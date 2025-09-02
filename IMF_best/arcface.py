from turtle import st
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
import numpy as np
##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      BatchNorm2d(64), 
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512), 
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
    
    def forward(self,x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class FaceSimLoss(nn.Module):
    def __init__(self, arcface_model, normalize=True, input_size=(112, 112)):
        """
        arcface_model: 预训练 ArcFace 模型，输出 shape=(B, embedding_dim)
        normalize: 是否对图像进行 [-1,1] -> [0,1] -> 标准化为 ArcFace 输入
        input_size: ArcFace 模型所需输入分辨率，默认 (112, 112)
        """
        super().__init__()
        self.arcface = arcface_model.eval()  # 固定权重
        self.arcface.requires_grad_(False)
        self.loss_fn = nn.CosineEmbeddingLoss()
        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

    def forward(self, fake_img, real_img):
        """
        fake_img: 生成图像，(B, 3, H, W)，值域 [-1, 1]
        real_img: 真实图像，(B, 3, H, W)，值域 [-1, 1]
        return: cosine embedding loss
        """
        # Resize 到 ArcFace 输入分辨率
        fake_img = F.interpolate(fake_img, size=(112, 112), mode='bilinear', align_corners=False)
        real_img = F.interpolate(real_img, size=(112, 112), mode='bilinear', align_corners=False)


        # Normalize to mean=0.5, std=0.5
        fake_img = (fake_img -self.mean) / self.std
        real_img = (real_img -self.mean) / self.std

        with torch.no_grad():
            feat_fake = self.arcface(fake_img)
            feat_real = self.arcface(real_img)

        target = torch.ones(feat_fake.size(0), device=fake_img.device)
        loss = self.loss_fn(feat_fake, feat_real, target)
        return loss
# --------- Main Entrypoint ----------
from torchvision import transforms
from PIL import Image
import os
# --------- Cosine Similarity Function ----------
def compare_images(model, img1: torch.Tensor, img2: torch.Tensor):
    model.eval()
    with torch.no_grad():
        input_batch = torch.stack([img1, img2])  # shape: (2,3,112,112)
        embeddings = model(input_batch)          # shape: (2,512)
        sim = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    return sim.item()
def main(img_path1, img_path2, model_path=None):
    # transform
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # load images
    img1 = transform(Image.open(img_path1).convert("RGB"))
    img2 = transform(Image.open(img_path2).convert("RGB"))

    # load model
    model = Backbone(50, 0.65, mode='ir_se')
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    model.eval()

    # compare
    similarity = compare_images(model, img1, img2)
    print(f"Cosine similarity between the two images: {similarity:.4f}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True, help="Path to first image")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image")
    parser.add_argument("--model", type=str, default=None, help="(Optional) path to pretrained ArcFace model")
    args = parser.parse_args()
    main(args.img1, args.img2, args.model)
