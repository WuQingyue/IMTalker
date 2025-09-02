from email.policy import strict
from pyexpat import model
from joblib import PrintTime
from sklearn.metrics import top_k_accuracy_score
import torch
import torch.nn as nn
from networks.model_bestdecoder import IMFModel
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from dataset import VFHQ_test
from torch.utils import data
import cv2
import torchvision.transforms as transforms

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def _load_image_256(path):
    """加载图像并转换为所需格式"""
    transform_256 = transforms.Compose([
        transforms.Resize((256, 256)),  # 将图像调整为 512x512
        transforms.ToTensor(),  # 转换为 Tensor
    ])
    image = Image.open(path).convert("RGB")
    image_tensor = transform_256(image)
    return image_tensor

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 1)
    vid = vid.clamp(-1, 1).cpu().numpy()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).astype(np.uint8)
    T, H, W, C = vid.shape

    #vid = np.concatenate((vid), axis=-2)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    # 写入每一帧
    for frame in vid:
        # OpenCV 要求输入为 BGR，因此需要从 RGB 转换
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Video saved successfully to {save_path}")
    #print(save_path)
    #torchvision.io.write_video(save_path, vid, fps=fps)

import lpips
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import csv
# 加载 LPIPS 计算模型
lpips_model = lpips.LPIPS(net='vgg').cuda()  # 使用 VGG 作为 backbone

def compute_ssim(img1, img2):
    """计算单张图像的 SSIM"""
    img1 = img1.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # 转换为 NumPy 格式
    img2 = img2.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    return ssim(img1_gray, img2_gray, data_range=1.0)  # 计算 SSIM

def compute_lpips(img1, img2):
    """计算单张图像的 LPIPS"""
    return lpips_model(img1, img2).item()  # LPIPS 计算

def compute_metrics(vid_target_recon, vid_source_recon, vid_target):
    """计算整个视频序列的 SSIM 和 LPIPS"""
    ssim_target_recon, ssim_source_recon = [], []
    lpips_target_recon, lpips_source_recon = [], []
    
    for i in range(vid_target.shape[0]):  
        img_target = vid_target[i].unsqueeze(0)  
        img_target_recon = vid_target_recon[i].unsqueeze(0)
        img_source_recon = vid_source_recon[i].unsqueeze(0)

        # 计算 SSIM
        ssim_target_recon.append(compute_ssim(img_target, img_target_recon))
        ssim_source_recon.append(compute_ssim(img_target, img_source_recon))

        # 计算 LPIPS
        lpips_target_recon.append(compute_lpips(img_target, img_target_recon))
        lpips_source_recon.append(compute_lpips(img_target, img_source_recon))

    # 计算均值
    return {
        "ssim_target_recon": np.mean(ssim_target_recon),
        "ssim_source_recon": np.mean(ssim_source_recon),
        "lpips_target_recon": np.mean(lpips_target_recon),
        "lpips_source_recon": np.mean(lpips_source_recon)
    }

def save_metrics_to_csv(video_id, metrics, save_path="metrics_results.csv"):
    """将计算出的 SSIM 和 LPIPS 结果保存到 CSV 文件"""
    file_exists = os.path.exists(save_path)

    with open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 如果文件不存在，则写入表头
        if not file_exists:
            writer.writerow(["video_id", "ssim_target_recon", "ssim_source_recon", "lpips_target_recon", "lpips_source_recon"])
        
        # 写入数据
        writer.writerow([video_id, metrics["ssim_target_recon"], metrics["ssim_source_recon"], metrics["lpips_target_recon"], metrics["lpips_source_recon"]])
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import imageio
def visualize_features(f_r, aligned_f, layer_idx=0, num_channels=3, title_prefix='', save_path="./viz_dir"):
    """
    f_r: 原始 reference 的 dense features（list of tensors）
    aligned_f: 对齐后的 features（list of tensors）
    layer_idx: 要可视化第几层（通常 0 是高分辨率，3 是低分辨率）
    num_channels: 显示前几个通道
    """

    # 取第 layer_idx 层的特征
    f_r_layer = f_r[layer_idx][0].detach().cpu()   # shape: [C, H, W]
    f_aligned_layer = aligned_f[layer_idx][0].detach().cpu()

    # 归一化 + 可视化
    def normalize_feature_map(fm):
        fm = fm - fm.min()
        fm = fm / (fm.max() + 1e-5)
        return fm

    fig, axs = plt.subplots(2, num_channels, figsize=(num_channels * 3, 6))

    for i in range(num_channels):
        # 原始特征
        axs[0, i].imshow(normalize_feature_map(f_r_layer[i]), cmap='viridis')
        axs[0, i].axis('off')
        axs[0, i].set_title(f'{title_prefix}Ref Ch{i}')

        # 对齐特征
        axs[1, i].imshow(normalize_feature_map(f_aligned_layer[i]), cmap='viridis')
        axs[1, i].axis('off')
        axs[1, i].set_title(f'{title_prefix}Aligned Ch{i}')

    axs[0, 0].set_ylabel('Before Alignment', fontsize=12)
    axs[1, 0].set_ylabel('After Alignment', fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        if args.model == 'vox':
            model_path = 'checkpoints/vox.pt'
        elif args.model == 'taichi':
            model_path = 'checkpoints/taichi.pt'
        elif args.model == 'ted':
            model_path = 'checkpoints/ted.pt'
        elif args.model == 'vfhq':
            model_path = args.model_path
        else:
            raise NotImplementedError

        print('==> loading model')
        self.gen = IMFModel().to("cuda")
        weight = torch.load(model_path, map_location=lambda storage, loc: storage )["state_dict"]
        ae_state_dict = {k.replace("gen.", ""): v for k, v in weight.items() if k.startswith("gen.")}
        self.gen.load_state_dict(ae_state_dict,strict=False)
        self.gen.eval()

        print('==> loading data')
        self.save_path = args.save_folder
        os.makedirs(self.save_path, exist_ok=True) 
        self.dataset_test = VFHQ_test(test_dir="E:\data\eval\self_reenactment\\vfhq", device="cuda")
        self.loader_test = data.DataLoader(
            self.dataset_test,
            num_workers=0,
            batch_size=1,
            sampler=None,
            pin_memory=False,
            drop_last=False,
        )
        #self.save_path = os.path.join(self.save_path, Path(args.source_path).stem + '_' + Path(args.driving_path).stem + '.mp4')
        #self.img_source = img_preprocessing(args.source_path, args.size).cuda()
        #self.vid_target, self.fps = vid_preprocessing(args.driving_path)
        #self.vid_target = self.vid_target.cuda()

    

#    def run(self):
#        print('==> running')
#        with torch.no_grad():
#            pbar = tqdm(range(len(self.dataset_test)), desc="Inferencing Progress")
#            loader = sample_data(self.loader_test)
#
#            for idx in pbar:
#                batch = next(loader)
#                source = batch["image_256"]
#                driving = batch["video_frames_256"]
#                video_id = batch["video_id"][0]
#
#                # 创建帧保存目录
#                layer_idx = 3
#                save_dir = os.path.join(self.save_path, f"{video_id}_{layer_idx}_features")
#                os.makedirs(save_dir, exist_ok=True)
#                image_paths = []
#
#                for i in tqdm(range(len(driving))):
#                    f_r = self.gen.encode_dense_feature(source)
#                    f_c = self.gen.encode_dense_feature(driving[i])
#                    t_r = self.gen.encode_latent_token(source)
#                    t_c = self.gen.encode_latent_token(driving[i])
#                    m_r = self.gen.decode_latent_token(t_r)
#                    m_c = self.gen.decode_latent_token(t_c)
#                    aligned_f = self.gen.ima(m_c, m_r, f_r)
#
#                    # 保存图像帧
#                    
#                    frame_path = os.path.join(save_dir, f"frame_{layer_idx}_{i:04d}.png")
#                    visualize_features(f_c, aligned_f, layer_idx=layer_idx, num_channels=3,
#                                       title_prefix=f"Frame {i}", save_path=frame_path)
#                    image_paths.append(frame_path)
#
#                # 合成视频
#                video_path = os.path.join(self.save_path, f"{video_id}_{layer_idx}_features.mp4")
#                with imageio.get_writer(video_path, fps=10) as writer:
#                    for img_path in image_paths:
#                        frame = imageio.imread(img_path)
#                        writer.append_data(frame)
#
#                print(f"Saved visualization video to {video_path}")

    def run(self):
        print('==> running')
        all_tokens = []  # 保存所有视频帧的 latent token（t_c）
    
        with torch.no_grad():
            pbar = tqdm(range(len(self.dataset_test)), desc="Inferencing Progress")
            loader = sample_data(self.loader_test)
    
            all_tokens = []
            all_video_ids = []
            
            for idx in pbar:
                batch = next(loader)
                source = batch["image_256"]
                driving = batch["video_frames_256"]
                video_id = batch["video_id"][0]
                # 创建帧保存目录
                layer_idx = 0
                save_dir = os.path.join(self.save_path, f"{video_id}_{layer_idx}_features")
                os.makedirs(save_dir, exist_ok=True)
                image_paths = []
                for i in range(len(driving)):
                    t_c = self.gen.encode_latent_token(driving[i])  # shape: [1, 32]
                    m_c = self.gen.latent_token_decoder(t_c)
                    f_c = self.gen.encode_dense_feature(driving[i])
                    frame_path = os.path.join(save_dir, f"frame_{layer_idx}_{i:04d}.png")
                    visualize_features(f_c, m_c, layer_idx=layer_idx, num_channels=3,
                                       title_prefix=f"Frame {i}", save_path=frame_path)
                    image_paths.append(frame_path)

                # 合成视频
                video_path = os.path.join(self.save_path, f"{video_id}_{layer_idx}_features.mp4")
                with imageio.get_writer(video_path, fps=10) as writer:
                    for img_path in image_paths:
                        frame = imageio.imread(img_path)
                        writer.append_data(frame)

                print(f"Saved visualization video to {video_path}")
                    #img_target_recon = self.gen(driving[i], source)
                        
                    #self.gen.implicit_motion_alignment[0].visualize_alignment(img_target_recon, save_path=attn_path)
                    #self.gen.frame_decoder.visualize_layers(img_target_recon, save_path=layer_path)
                    # 读取并合并可视化图
                    #attn_img = cv2.imread(attn_path)
                    #layer_img = cv2.imread(layer_path)
                    #att_frames.append(attn_img)
                    #layer_frames.append(layer_img)                
                    #img_source_recon = self.gen(batch['images_256'][i], batch['images_256'][i])
                    
                    
                    #vid_target_recon.append(img_target_recon)
                    #vid_source_recon.append(img_source_recon)
                    #os.remove(attn_path)  # 删除临时文件
                    #os.remove(layer_path)  # 删除临时文件
                ## 保存可视化视频
                #if att_frames:
                #    h, w, _ = att_frames[0].shape
                #    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                #    writer = cv2.VideoWriter(output_att_path, fourcc, 10, (w, h))
                #    for frame in att_frames:
                #        writer.write(frame)
                #    writer.release()
#
                ## 保存可视化视频
                #if layer_frames:
                #    h, w, _ = layer_frames[0].shape
                #    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                #    writer = cv2.VideoWriter(output_layer_path, fourcc, 10, (w, h))
                #    for frame in layer_frames:
                #        writer.write(frame)
                #    writer.release()
                #vid_target_recon = torch.cat(vid_target_recon, dim=0)
                #vid_source_recon = torch.cat(vid_source_recon, dim=0)
                #vid_target = torch.cat(batch['images_512'], dim=0)
                # 计算 SSIM 和 LPIPS
                #metrics = compute_metrics(vid_target_recon, vid_source_recon, vid_target)

                # 保存到 CSV
                #save_metrics_to_csv(video_id, metrics)

                # 打印评估结果
                #print(f"Video {video_id} - SSIM (target recon): {metrics['ssim_target_recon']:.4f}, SSIM (source recon): {metrics['ssim_source_recon']:.4f}")
                #print(f"Video {video_id} - LPIPS (target recon): {metrics['lpips_target_recon']:.4f}, LPIPS (source recon): {metrics['lpips_source_recon']:.4f}")
                #save_video(vid_target_recon, os.path.join(self.save_path, batch['video_id'][0]+'.mp4'), 25)


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted', 'vfhq'], default='vfhq')
    parser.add_argument("--save_folder", type=str, default='E:\\codes\\codes\\LIA\\result_vfhq_381000\\')
    parser.add_argument("--model_path", type=str, default="E:\\codes\\codes\\LIA\\a100_rvm\\log\\ckpt\\381000.pt")
    args = parser.parse_args()

    # demo
    demo = Demo(args)
    demo.run()
