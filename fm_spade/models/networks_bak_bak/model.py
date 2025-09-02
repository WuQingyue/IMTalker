from pickle import FALSE
from regex import P
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.spectral_norm as spectral_norm
from models.utils.modules import FeatResBlock,UpConvResBlock,DownConvResBlock,ResBlock, SPADEResnetBlock
from models.utils.lia_resblocks import StyledConv,EqualConv2d,EqualLinear
from models.utils.vit import ImplicitMotionAlignment

import math

class DenseFeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, output_channels=[128, 256, 512, 512], initial_channels=64):
        super().__init__()
        
        # Initial convolution layer with BatchNorm and ReLU
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(initial_channels, track_running_stats=True, eps=1e-4),  # Stable eps for better numerical stability
            nn.ReLU(inplace=True)
        )
        
        # List of downsampling blocks (DownConvResBlocks)
        self.down_blocks = nn.ModuleList()
        current_channels = initial_channels
        
        # Initial block that doesn't change the number of channels
        self.down_blocks.append(DownConvResBlock(current_channels, current_channels))
        
        # Add down blocks for each specified output channel
        for out_channels in output_channels:
            self.down_blocks.append(DownConvResBlock(current_channels, out_channels))
            current_channels = out_channels

    def forward(self, x):
        features = []
        x = self.initial_conv(x)
        
        # Apply downsampling blocks and collect features from the second block onwards
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i >= 1:  # Collect features from the second block onwards
                features.append(x)
        
        return features


class LatentTokenEncoder(nn.Module):
    def __init__(self, initial_channels=64, output_channels=[64, 128, 256, 512, 512, 512], dm=32):
        super(LatentTokenEncoder, self).__init__()

        # Initial convolution followed by LeakyReLU activation
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)

        # Dynamically create ResBlocks
        self.res_blocks = nn.ModuleList()
        in_channels = initial_channels
        for out_channels in output_channels:
            self.res_blocks.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels

        # Equal convolution and linear layers
        self.equalconv = EqualConv2d(output_channels[-1], output_channels[-1], kernel_size=3, stride=1, padding=1)
        self.linear_layers = nn.ModuleList([EqualLinear(output_channels[-1], output_channels[-1]) for _ in range(4)])
        self.final_linear = EqualLinear(output_channels[-1], dm)

    def forward(self, x):
        # Initial convolution and activation
        x = self.activation(self.conv1(x))
        
        # Apply ResBlocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply equalconv
        x = self.equalconv(x)
        
        # Global average pooling
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2)
        
        # Apply linear layers
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))
        
        # Final linear layer
        x = self.final_linear(x)
        
        return x


class LatentTokenDecoder(nn.Module):
    def __init__(self, latent_dim=32, const_dim=32):
        super().__init__()
        # Constant input for the decoder
        self.const = nn.Parameter(torch.randn(1, const_dim, 4, 4))
        
        # StyleConv layers
        self.style_conv_layers = nn.ModuleList([
            StyledConv(const_dim, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 256, 3, latent_dim, upsample=True),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 128, 3, latent_dim, upsample=True),
            StyledConv(128, 128, 3, latent_dim),
            StyledConv(128, 128, 3, latent_dim)  
        ])

    def forward(self, t):
        # Repeat constant input for batch size
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        
        # Store feature maps
        m1, m2, m3, m4 = None, None, None, None
        
        # Apply style convolution layers
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            if i == 3:
                m1 = x
            elif i == 6:
                m2 = x
            elif i == 9:
                m3 = x
            elif i == 12:
                m4 = x
        
        # Return the feature maps in reverse order
        return m4, m3, m2, m1

    
class FrameDecoder(nn.Module):
    def __init__(self, upscale=1):
        super().__init__()
        self.upscale = upscale
        # Upsampling blocks
        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(512, 512),
            UpConvResBlock(1024, 512),
            UpConvResBlock(768, 256),
            UpConvResBlock(384, 128),
            UpConvResBlock(128, 64),
        ])
        
        # Feature blocks (repeated FeatResBlock for each resolution)
        self.feat_blocks = nn.ModuleList([
            nn.Sequential(*[FeatResBlock(512)] * 3),
            nn.Sequential(*[FeatResBlock(256)] * 3),
            nn.Sequential(*[FeatResBlock(128)] * 3),
        ])
        
        # Final output layer
        if self.upscale is None or self.upscale <= 0:
            self.final_conv = nn.Sequential(
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.final_conv_512 = nn.Sequential(
                nn.Conv2d(64, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.Sigmoid()
            ) 

    def forward(self, features):
        # Reshape features if necessary and prepare for decoding
        reshaped_features = [
            feat.view(*feat.shape[:2], int(math.sqrt(feat.shape[1])), -1) if len(feat.shape) == 3 else feat
            for feat in features
        ]
        x = reshaped_features[-1]  # Start with the smallest feature map
        for i, upconv_block in enumerate(self.upconv_blocks):
            x = upconv_block(x)  # Apply upsampling block
            if i < len(self.feat_blocks):
                feat_input = reshaped_features[-(i + 2)]
                x = torch.cat([x, self.feat_blocks[i](feat_input)], dim=1)
        if self.upscale is None or self.upscale <= 0:
            return self.final_conv(x)
        else:
            return self.final_conv_512(x)

class IMFModel(nn.Module):
    '''
    IMFModel consists of the following components:
    - DenseFeatureEncoder (EF): Encodes the reference frame into multi-scale features.
    - LatentTokenEncoder (ET): Encodes both current and reference frames into latent tokens.
    - LatentTokenDecoder (IMFD): Decodes latent tokens into motion features.
    - ImplicitMotionAlignment (IMFA): Aligns reference features to the current frame using motion features.
    '''

    def __init__(self, upscale=1):
        super().__init__()
        self.latent_token_encoder = LatentTokenEncoder(initial_channels=64, output_channels=[128, 256, 512, 512, 512], dm=32)
        self.latent_token_decoder = LatentTokenDecoder()

        self.feature_dims = [128, 256, 512, 512]
        self.spatial_dims = [(64, 64), (32, 32), (16, 16), (8, 8)]
        self.motion_dims = [128, 256, 512, 512]

        self.dense_feature_encoder = DenseFeatureEncoder(output_channels=self.feature_dims)

        # Initialize ImplicitMotionAlignment modules
        self.implicit_motion_alignment = nn.ModuleList(
            [ImplicitMotionAlignment(f, m, s) for f, m, s in zip(self.feature_dims, self.motion_dims, self.spatial_dims)]
        )

        self.frame_decoder = FrameDecoder(upscale)

    def encode_dense_feature(self, x_reference):
        f_r = self.dense_feature_encoder(x_reference)
        return f_r

    def encode_latent_token(self, x_reference):
        t_c = self.latent_token_encoder(x_reference)
        return t_c

    def tokens(self, x_current, x_reference):
        f_r = self.dense_feature_encoder(x_reference)
        t_r, t_c = self.latent_token_encoder(x_reference), self.latent_token_encoder(x_current)
        return f_r, t_r, t_c

    def decode_latent_tokens(self, f_r, t_r, t_c):
        m_c, m_r = self.latent_token_decoder(t_c), self.latent_token_decoder(t_r)
        aligned_features = [
            align_layer(m_c_i, m_r_i, f_r_i)  # 传递 mask_i
            for m_c_i, m_r_i, f_r_i,align_layer in zip(m_c, m_r, f_r,  self.implicit_motion_alignment)
        ]
        return self.frame_decoder(aligned_features)

    def ima(self, m_c, m_r, f_r):
        aligned_features = [
            align_layer(m_c_i, m_r_i, f_r_i)  # 传递 mask_i
            for m_c_i, m_r_i, f_r_i,align_layer in zip(m_c, m_r, f_r,  self.implicit_motion_alignment)
        ]
        return self.frame_decoder(aligned_features)

    def forward(self, x_current, x_reference):
        f_r, t_r, t_c = self.tokens(x_current, x_reference)

        m_c, m_r = self.latent_token_decoder(t_c), self.latent_token_decoder(t_r)   

        aligned_features = [
        align_layer(m_c_i, m_r_i, f_r_i)  # 传递 mask_i
        for m_c_i, m_r_i, f_r_i,align_layer in zip(m_c, m_r, f_r, self.implicit_motion_alignment)
        ]
        
        return self.frame_decoder(aligned_features)
    
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import time

    # 假设你已经定义了 IMFModel 及其依赖的模块
    model = IMFModel().cuda()
    model.eval()

    # 打印各模块参数量
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print("🔍 模块参数统计：")
    for name, submodule in model.named_children():
        print(f"{name:30s}: {count_parameters(submodule):,} 参数")

    print(f"{'全部模型':30s}: {count_parameters(model):,} 参数")

    # 随机输入测试
    x_current = torch.randn(1000, 3, 256, 256).cuda()
    x_reference = torch.randn(1, 3, 256, 256).cuda()
    mask_list = [ torch.randn(1, 256, 256).cuda()]

    # 前向传播耗时
    print("\n⏱️ 正在运行 1000 次前向传播...")
    start_time = time.time()

    with torch.no_grad():
        f_r = model.encode_dense_feature(x_reference)
        t_r = model.encode_latent_token(x_reference)
        for i in range(1000):
            x = x_current[i].unsqueeze(dim=0)
            t_c = model.encode_latent_token(x)
            _ = model.decode_latent_tokens(f_r, t_r, t_c, mask_list)

    end_time = time.time()
    print(f"✅ 完成！总耗时: {end_time - start_time:.2f} 秒，平均每次: {(end_time - start_time)/1000:.4f} 秒")
