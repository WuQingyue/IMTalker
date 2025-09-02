from pickle import FALSE
from regex import P
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.spectral_norm as spectral_norm
from utils.modules import FeatResBlock,UpConvResBlock,DownConvResBlock,ResBlock, SPADEDecoder
from utils.lia_resblocks import StyledConv,EqualConv2d,EqualLinear
from utils.vit import ImplicitMotionAlignment
import math
from utils.modules import ConvResBlock
from utils.vit import TransformerBlock, CrossAttentionModule
import torch
import torch.nn as nn
# 假设 EqualConv2d, EqualLinear, DownConvResBlock 都是您项目中已定义的层

class DenseFeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, output_channels=[128, 256, 512, 512], initial_channels=64, dm=512):
        super().__init__()
        
        # 初始卷积层和下采样模块 (保持不变)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList()
        current_channels = initial_channels
        self.down_blocks.append(DownConvResBlock(current_channels, current_channels))
        for out_channels in output_channels:
            self.down_blocks.append(DownConvResBlock(current_channels, out_channels))
            current_channels = out_channels

        # --- 处理器和融合网络的定义 (与上一版相同，为所有6层特征做准备) ---
        all_feature_channels = [initial_channels] + [b.conv2.out_channels for b in self.down_blocks]
        common_dim = output_channels[-1]
        
        self.feature_processors = nn.ModuleList()
        for in_ch in all_feature_channels:
            self.feature_processors.append(EqualLinear(in_ch, common_dim))
        
        num_features = len(all_feature_channels)
        self.weighting_net = nn.Sequential(
            nn.Linear(num_features * common_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_features)
        )

        self.final_projection = EqualLinear(common_dim, dm)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        # --- 核心修改点: 分别收集用于计算的特征和用于返回的特征 ---
        
        # 列表1: 用于内部计算，包含所有层级的特征
        features_for_vector_fusion = []
        # 列表2: 用于最终返回，只包含约定的深层特征
        features_to_return = []
        
        # 初始卷积层
        x = self.initial_conv(x)
        features_for_vector_fusion.append(x) # 存入内部列表
        
        # 下采样模块
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            features_for_vector_fusion.append(x) # 每一层都存入内部列表
            
            if i >= 1:
                # 仅当满足原始条件时，才存入返回列表
                features_to_return.append(x) 

        # --- 语义向量的计算逻辑不变，使用包含所有信息的内部列表 ---
        semantic_vectors = []
        for feature_map, processor in zip(features_for_vector_fusion, self.feature_processors):
            pooled_vec = feature_map.mean(dim=[2, 3]) 
            semantic_vectors.append(self.activation(processor(pooled_vec)))

        stacked_vectors = torch.stack(semantic_vectors, dim=1)
        concatenated_vectors = stacked_vectors.view(x.size(0), -1)
        weights = self.weighting_net(concatenated_vectors).softmax(dim=1)
        weighted_sum = (weights.unsqueeze(-1) * stacked_vectors).sum(dim=1)
        final_semantic_feature = self.final_projection(weighted_sum)
        # --- 最终返回 ---
        # 返回约定的特征图列表 和 基于全部信息融合的语义向量
        return features_to_return, final_semantic_feature, stacked_vectors


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
    def __init__(self, latent_dim=544, const_dim=32):
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

    def forward(self, t, f):
        # Repeat constant input for batch size
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        #import pdb;pdb.set_trace()
        # Store feature maps
        m1, m2, m3, m4 = None, None, None, None
        t = torch.concat((t,f), dim=-1)
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
    def __init__(self, feature_dims, spatial_dims, depth):
        """
        一个健壮的、支持双特征流融合的解码器。
        
        Args:
            feature_dims (list): 编码器输出的各尺度特征通道数，从浅到深。
                                 例如: [64, 128, 256, 512]
            spatial_dims (list): 编码器输出的各尺度特征空间尺寸，从浅到深。
                                 例如: [128, 64, 32, 16]
            depth (int): 每个尺度上TransformerBlock的重复次数。
        """
        super().__init__()
        
        # 反转列表，方便由深到浅进行索引 (0=最深层)
        feature_dims = feature_dims[::-1]
        spatial_dims = spatial_dims[::-1]
        
        # ------------------ 模块定义区 ------------------

        # 1. 用于处理最深层融合的初始模块
        #    输入: src最深层 + align最深层
        self.conv_in = ConvResBlock(feature_dims[0], feature_dims[0])

        # 2. 上采样模块列表
        #    将特征从深层(i)上采样至下一层(i+1)
        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(feature_dims[i], feature_dims[i+1]) for i in range(len(feature_dims) - 1)
        ])

        # 3. 卷积残差模块列表
        #    用于处理融合后的三股特征
        #    输入: 上采样特征 + src特征 + align特征
        self.resblocks = nn.ModuleList([
            ConvResBlock(feature_dims[i+1] * 2, feature_dims[i+1]) for i in range(len(feature_dims) - 1)
        ])
        
        # 4. Transformer模块列表
        #    用于在每个尺度上进行特征增强
        self.transformer_blocks = nn.ModuleList([
            # 最深层 (循环外处理)
            nn.Sequential(*[TransformerBlock(dim=feature_dims[0], heads=8, dim_spatial=spatial_dims[0]**2, mlp_dim=1024) for _ in range(depth)]),
            # 后续层 (循环内处理)
            *[nn.Sequential(*[TransformerBlock(dim=feature_dims[i+1], heads=8, dim_spatial=spatial_dims[i+1]**2, mlp_dim=1024) for _ in range(depth)]) for i in range(len(feature_dims) - 1)]
        ])

        # 5. 最终输出层
        #self.final_conv = nn.Sequential(
        #    UpConvResBlock(feature_dims[-1], 64),
        #    UpConvResBlock(64, 32),
        #    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #    nn.Conv2d(32, 3, kernel_size=3, padding=1),
        #    nn.Sigmoid(),
        #)
        self.final_conv = SPADEDecoder(upscale=2, max_features=128)

    def forward(self, features_align):
        """
        Args:
            features_src (list): 源编码器的特征列表，从浅到深。
            features_align (list): 对齐编码器的特征列表，从浅到深。
        """
        
        # ------------------ 前向传播区 ------------------

        # 1. 处理最深层 (解码起点)
        #    融合src和align的最深层特征
        x = features_align[-1]

        x = self.conv_in(x)
        x = self.transformer_blocks[0](x) # 对最深层的融合结果应用Transformer

        # 2. 循环处理后续层，由深到浅
        #    循环次数 = 上采样模块的数量
        for i in range(len(self.upconv_blocks)):
            
            # (a) 上采样
            x = self.upconv_blocks[i](x)
            
            # (b) 融合三股特征：上采样特征、src特征、align特征
            #     -i-2 会从列表末尾开始依次向前取值，例如-2, -3, -4...
            align_skip = features_align[-(i + 2)]
            x = torch.cat([x, align_skip], dim=1)
            
            # (c) 通过ResBlock进行深度融合
            x = self.resblocks[i](x)
            
            # (d) 通过Transformer进行全局特征增强
            #     transformer_blocks[0]已用于最深层，所以这里从[1]开始
            x = self.transformer_blocks[i + 1](x)
            
        # 3. 通过最终输出层生成图像
        out = self.final_conv(x)
        return out

class IMFModel(nn.Module):
    '''
    IMFModel consists of the following components:
    - DenseFeatureEncoder (EF): Encodes the reference frame into multi-scale features.
    - LatentTokenEncoder (ET): Encodes both current and reference frames into latent tokens.
    - LatentTokenDecoder (IMFD): Decodes latent tokens into motion features.
    - ImplicitMotionAlignment (IMFA): Aligns reference features to the current frame using motion features.
    '''

    def __init__(self, args):
        super().__init__()
        self.latent_token_encoder = LatentTokenEncoder(initial_channels=64, output_channels=[128, 256, 512, 512, 512])
        self.latent_token_decoder = LatentTokenDecoder()

        self.feature_dims = [128, 256, 512, 512]
        self.spatial_dims = [64, 32, 16, 8]
        self.motion_dims = [128, 256, 512, 512]

        self.dense_feature_encoder = DenseFeatureEncoder(output_channels=self.feature_dims)

        # Initialize ImplicitMotionAlignment modules
        self.implicit_motion_alignment = nn.ModuleList(
            [CrossAttentionModule(dim_spatial=s * s, dim_qk=m, dim_v=f) for s, m, f in zip(self.spatial_dims, self.motion_dims, self.feature_dims)]
        )

        self.frame_decoder = FrameDecoder(self.feature_dims, self.spatial_dims, args.depth)

    def encode_dense_feature(self, x_reference):
        f_r = self.dense_feature_encoder(x_reference)
        return f_r

    def encode_latent_token(self, x_reference):
        t_c = self.latent_token_encoder(x_reference)
        return t_c

    def tokens(self, x_current, x_reference):
        f_r,f, f_emb = self.dense_feature_encoder(x_reference)
        t_r, t_c = self.latent_token_encoder(x_reference), self.latent_token_encoder(x_current)
        return f_r, t_r, t_c, f, f_emb

    def decode_latent_tokens(self, f_r, t_r, t_c, f):
        m_c, m_r = self.latent_token_decoder(t_c, f), self.latent_token_decoder(t_r, f)
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
        f_r, t_r, t_c, f, f_emb = self.tokens(x_current, x_reference)

        m_c, m_r = self.latent_token_decoder(t_c, f), self.latent_token_decoder(t_r, f)   

        aligned_features = [
        align_layer(m_c_i, m_r_i, f_r_i)  # 传递 mask_i
        for m_c_i, m_r_i, f_r_i,align_layer in zip(m_c, m_r, f_r, self.implicit_motion_alignment)
        ]
        
        return self.frame_decoder(aligned_features)
    
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import time
    import argparse
    args = argparse.Namespace()
    args.depth = 4
    # 假设你已经定义了 IMFModel 及其依赖的模块
    model = IMFModel(args).cuda()
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
            _ = model.decode_latent_tokens(f_r, t_r, t_c)

    end_time = time.time()
    print(f"✅ 完成！总耗时: {end_time - start_time:.2f} 秒，平均每次: {(end_time - start_time)/1000:.4f} 秒")
