from torch import nn
import torch
from torchvision import models
import numpy as np
import torch.nn.functional as F
class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out
class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):

        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)

        return out_dict


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        vgg_model = models.vgg19(pretrained=True)
        # vgg_model.load_state_dict(torch.load('./vgg19-dcbb9e9d.pth'))
        vgg_pretrained_features = vgg_model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):

        X = X.clamp(-1, 1)
        X = X / 2 + 0.5
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return out


class VGGLoss_mask(nn.Module):
    def __init__(self, device="cuda"):
        super(VGGLoss_mask, self).__init__()
        self.device = device
        self.scales = [1.0, 0.5, 0.25, 0.125] # 您的原始尺度
        
        # 将模块移动到指定设备
        self.pyramid = ImagePyramide(self.scales, 3).to(self.device)
        self.vgg = Vgg19().to(self.device) # 假设Vgg19也接受device参数
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0) # 权重设为1更常见

    def forward(self, img_recon, img_real, mask):
        # 使用您自己的高质量金字塔来处理图像
        pyramid_real = self.pyramid(img_real)
        pyramid_recon = self.pyramid(img_recon)

        loss_all = 0.0
        loss_face = 0.0
        
        # 外层循环：遍历每个图像尺度
        for scale in self.scales:
            scale_str = str(scale)
            scale_key = f'prediction_{scale_str}'

            # 步骤 1: 为蒙版进行独立的、逻辑对应的降采样
            # 这是最关键的修正：在进入内层循环前，先准备好当前尺度的蒙版
            if scale == 1.0:
                mask_at_scale = mask
            else:
                # 使用与您的 AntiAliasInterpolation2d 的降采样部分最接近的方法
                mask_at_scale = F.interpolate(mask, scale_factor=scale, mode='nearest', recompute_scale_factor=True)

            # 获取当前尺度的VGG特征
            recon_feats = self.vgg(pyramid_recon[scale_key])
            real_feats = self.vgg(pyramid_real[scale_key])

            # 内层循环：遍历VGG特征层
            for i, weight in enumerate(self.weights):
                feat_real = real_feats[i].detach()
                feat_recon = recon_feats[i]

                # --- 全局损失 ---
                all_loss_i = torch.abs(feat_recon - feat_real).mean()
                loss_all += all_loss_i * weight
                
                # --- 面部损失 ---
                # 步骤 2: 使用已经按尺度缩放好的 mask_at_scale
                mask_i = F.interpolate(mask_at_scale, size=feat_real.shape[2:], mode='nearest')
                
                diff_mask = torch.abs((feat_recon - feat_real) * mask_i)
                
                mask_sum = mask_i.sum()
                if mask_sum > 1e-6:
                    mask_loss_i = diff_mask.sum() / mask_sum
                    loss_face += mask_loss_i * weight
        
        return loss_all, loss_face


if __name__ == "__main__":
    import pickle
    from PIL import Image
    from torchvision import transforms
    import os 
    from glob import glob
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    vgg = VGGLoss()
    path = "E:\\data\\vfhq_flame\\Clip+_5rNOD1nsDg+P0+C0+F575-769\\optim.pkl"
    with open(path, 'rb') as f:
        data = pickle.load(f)
    imgs_path = "E:\\data\\vfhq_video_frame\\Clip+_5rNOD1nsDg+P0+C0+F575-769"
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 将图像调整为 512x512
        transforms.ToTensor()  # 转换为 Tensor
    ])
    frames = sorted(glob(os.path.join(imgs_path, "*.png")))
    for frame in frames:
        image = Image.open(frame).convert("RGB")
        image = transform(image)
        bbox = torch.tensor(data[os.path.basename(frame)]['bbox']*512).long()
        image = image[:,bbox[1]:bbox[3], bbox[0]:bbox[2]].unsqueeze(dim=0).cuda()
        pred = torch.zeros_like(image).cuda()
        def _resize(frames, tgt_size):
            frames = nn.functional.interpolate(
                frames, size=(tgt_size, tgt_size), mode='bilinear', align_corners=False, antialias=True
            )
            return frames
        loss = vgg(_resize(image, 512), _resize(pred, 512))

        print(loss)
        numpy_image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        # 使用 matplotlib 显示图片
        plt.imshow(numpy_image)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
