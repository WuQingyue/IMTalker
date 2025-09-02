import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class GuidedResampler(nn.Module):
    """
    引导式稀疏重采样模块 (V3: 显存优化)。

    该模块利用一个低分辨率的注意力图谱 (coarse_attn_map) 来指导在高分辨率的
    值(V)特征上进行稀疏采样，并通过对采样点的特征进行加权求和来重组特征，
    权重来源于粗糙注意力图谱，从而高效地实现特征的扭曲对齐。
    此版本通过向量化操作显著降低了显存占用。
    """

    def __init__(self, dim, downsample_ratio=4, k_top_samples=1):
        """
        初始化函数。
        Args:
            dim (int): 特征维度。
            downsample_ratio (int): 高分辨率与低分辨率特征图的尺寸比例。
            k_top_samples (int): 从低分辨率注意力图谱中为每个查询点选择的最相关关键点的数量。
        """
        super().__init__()
        self.dim = dim
        self.ratio = downsample_ratio
        self.k_samples = k_top_samples

    def forward(self, v_high_feat, coarse_attn_map):
        """
        前向传播函数。
        Args:
            v_high_feat (Tensor): 高分辨率的值特征 (B, C, H, W)。
            coarse_attn_map (Tensor): 来自低分辨率的注意力图谱 (B, N_low, N_low)。
        Returns:
            Tensor: 经过稀疏重采样和加权聚合后扭曲对齐的高分辨率特征 (B, C, H, W)。
        """
        # --- 1. 准备工作：获取维度信息并将特征图转换为序列 ---
        B, C, H, W = v_high_feat.shape
        H_low, W_low = H // self.ratio, W // self.ratio
        N_high = H * W
        N_low = H_low * W_low

        assert coarse_attn_map.shape == (B, N_low, N_low), \
            f"Coarse map shape mismatch. Expected {(B, N_low, N_low)}, but got {coarse_attn_map.shape}"

        v_high_seq = v_high_feat.flatten(2).transpose(1, 2)

        # --- 2. 从粗糙注意力图谱中找到 Top-K 索引和对应的权重 ---
        topk_values, topk_indices_low = torch.topk(coarse_attn_map, k=self.k_samples, dim=-1)

        # --- 3. 将低分辨率索引映射到高分辨率索引 (逻辑保持不变) ---
        topk_indices_low_row = topk_indices_low // W_low
        topk_indices_low_col = topk_indices_low % W_low

        topk_indices_high_topleft_row = topk_indices_low_row * self.ratio
        topk_indices_high_topleft_col = topk_indices_low_col * self.ratio
        
        delta = torch.stack(torch.meshgrid(
            torch.arange(self.ratio, device=v_high_feat.device),
            torch.arange(self.ratio, device=v_high_feat.device),
            indexing='ij'
        ), dim=-1).view(-1, 2)

        topleft = torch.stack([topk_indices_high_topleft_row, topk_indices_high_topleft_col], dim=-1)
        sparse_indices_2d = topleft.unsqueeze(-2) + delta.view(1, 1, 1, -1, 2)

        sparse_indices_1d = sparse_indices_2d[..., 0] * W + sparse_indices_2d[..., 1]
        sparse_indices_1d = sparse_indices_1d.view(B, N_low, -1)

        # --- 4. 为每个高分辨率查询点分配其对应的稀疏索引 (显存优化) ---
        # 原始方法使用 list comprehension，这里改为向量化操作
        high_res_q_coords = torch.stack(torch.meshgrid(
            torch.arange(H, device=v_high_feat.device),
            torch.arange(W, device=v_high_feat.device),
            indexing='ij'
        ), dim=-1).view(-1, 2)
        
        low_res_grid_indices = (high_res_q_coords[:, 0] // self.ratio) * W_low + (high_res_q_coords[:, 1] // self.ratio)
        
        # 将 low_res_grid_indices 扩展维度以用于 gather
        # (N_high) -> (B, N_high, 1) -> (B, N_high, K_sparse_len)
        K_sparse_len = sparse_indices_1d.shape[-1]
        low_res_grid_indices_expanded = low_res_grid_indices.view(1, N_high, 1).expand(B, -1, K_sparse_len)
        
        # 使用 gather 高效地选取索引
        final_sparse_indices = torch.gather(sparse_indices_1d, 1, low_res_grid_indices_expanded)

        # --- 5. 使用稀疏索引采集 Value 并进行加权聚合 (显存优化) ---
        # 5.1. 从完整的 v_high_seq 中采集数据 (显存优化)
        # 原始方法使用一个巨大的 expand 操作，这里改为更高效的高级索引
        batch_indices = torch.arange(B, device=v_high_feat.device).view(B, 1, 1)
        # 使用高级索引直接从 v_high_seq 中选取，避免创建巨大中间张量
        # v_sparse_seq: (B, N_high, K_sparse_len, C)
        v_sparse_seq = v_high_seq[batch_indices, final_sparse_indices]
        
        # 5.2. 准备权重 (显存优化)
        normalized_weights_low = F.softmax(topk_values, dim=-1)

        # 使用与上面类似的 gather 方法高效地选取权重
        low_res_grid_indices_weights_expanded = low_res_grid_indices.view(1, N_high, 1).expand(B, -1, self.k_samples)
        weights_high = torch.gather(normalized_weights_low, 1, low_res_grid_indices_weights_expanded)

        # 5.3. 执行加权聚合
        v_reshaped = v_sparse_seq.view(B, N_high, self.k_samples, self.ratio**2, C)
        v_block_mean = v_reshaped.mean(dim=3)
        weights_expanded = weights_high.unsqueeze(-1)
        warped_seq = (v_block_mean * weights_expanded).sum(dim=2)

        # --- 6. Reshape 输出到图像格式 ---
        warped_feat = warped_seq.transpose(1, 2).view(B, C, H, W)
        
        return warped_feat