import os, math, torch
import torch.nn as nn


class TransformerBlock(nn.Module):
	def __init__(self, dim, heads, dim_clip, mlp_dim):
		super().__init__()
		self.attention = nn.MultiheadAttention(dim, heads)
		self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)

	
	def forward(self, x):
		B, L, C = x.shape
		x_norm = self.norm1(x)
        # 修改此处获取注意力权重
		att_output, attn_weights = self.attention(x_norm, x_norm, x_norm)
		x_reshaped = x + att_output
		ff_output = self.mlp(self.norm2(x_reshaped))
		x_reshaped = x_reshaped + ff_output
		
		output = x_reshaped
        # 保存原始注意力矩阵（包含多头信息）
        #self.att = attn_weights.detach().cpu()  # shape: (num_heads, seq_len, seq_len)
		return output

class Audio_projection(nn.Module):
	def __init__(self, audio_dim, out_dim, clip_dim=60, depth=4, heads=8, mlp_dim=512, pose_dim=6):
		super().__init__()
		self.transformer_blocks = nn.ModuleList([TransformerBlock(audio_dim, heads, clip_dim, mlp_dim) for _ in range(depth)])
		self.projector = nn.Sequential(
            nn.Linear(audio_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, pose_dim)
        )
		self.injector = nn.Sequential(
            nn.Linear(pose_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, audio_dim)
        )
		self.mapping =  nn.Sequential(
            nn.Linear(audio_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, out_dim)
        )
		self.audio_dim = audio_dim
		self.clip_dim = clip_dim
	
	def forward(self, V_prime, pose=None):
		for i, block in enumerate(self.transformer_blocks):
			V_prime  = block(V_prime)
		project = self.projector(V_prime)
		if pose is not None:
			inject = self.injector(pose)
		else:
			inject = self.injector(project)
		return self.mapping(V_prime + inject), project
