# 文件名: attention_modules.py (格式统一版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_


# ============================================================================
# Part 1: 辅助函数
# ============================================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ============================================================================
# Part 2: 核心注意力模块
# ============================================================================

class StandardUnifiedAttention(nn.Module):
    """统一的标准注意力核心 (非窗口化)。"""

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, mask=None):
        """The forward pass now returns a tuple: (output_tensor, attention_map)"""
        B, N, C = query.shape

        # Project and reshape Q, K, V for multi-head attention
        q = self.q_proj(query).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Manual calculation to expose attention map
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn_map = attn.softmax(dim=-1)
        attn_map_dropped = self.attn_drop(attn_map)

        x = (attn_map_dropped @ v).transpose(1, 2).reshape(B, N, C)

        # Final projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_map

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

        # 5.3. 执行加权聚合 (无块内平均)
        # v_reshaped: (B, N_high, k_samples, ratio**2, C)
        v_reshaped = v_sparse_seq.view(B, N_high, self.k_samples, self.ratio**2, C)

        # 准备权重用于广播。
        # 将块级别的权重 (B, N_high, k_samples) 变形，并进行归一化，
        # 以便在像素级别上进行加权求和后，总权重为1。
        # weights_for_broadcast: (B, N_high, k_samples, 1, 1)
        weights_for_broadcast = (weights_high / (self.ratio**2)).view(B, N_high, self.k_samples, 1, 1)

        # 使用广播将权重应用到每个采样块内的所有像素，然后对所有采样点求和
        warped_seq = (v_reshaped * weights_for_broadcast).sum(dim=(2, 3))

        # --- 6. Reshape 输出到图像格式 ---
        warped_feat = warped_seq.transpose(1, 2).view(B, C, H, W)
        
        return warped_feat
class GuidedSparseAttention(nn.Module):
    """引导式稀疏注意力模块。"""

    def __init__(self, dim, num_heads, downsample_ratio=4, k_top_samples=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ratio = downsample_ratio
        self.k_samples = k_top_samples
        self.fine_attn_weights = StandardUnifiedAttention(dim, num_heads)

    def forward(self, q_high_feat, k_high_feat, v_high_feat, coarse_attn_map):
        B, C, H, W = q_high_feat.shape
        H_low, W_low = H // self.ratio, W // self.ratio
        N_high_q = H * W
        N_low_q = H_low * W_low
        N = H*W
        #import pdb;pdb.set_trace()
        assert coarse_attn_map.shape[:3] == (B, N_low_q, N_low_q), "Coarse map shape mismatch"

        q_high_seq = q_high_feat.flatten(2).transpose(1, 2)
        k_high_seq = k_high_feat.flatten(2).transpose(1, 2)
        v_high_seq = v_high_feat.flatten(2).transpose(1, 2)

        _, topk_indices_low = torch.topk(coarse_attn_map, k=self.k_samples, dim=-1)

        topk_indices_low_row = topk_indices_low // W_low
        topk_indices_low_col = topk_indices_low % W_low
        topk_indices_high_topleft_row = topk_indices_low_row * self.ratio
        topk_indices_high_topleft_col = topk_indices_low_col * self.ratio

        delta = torch.stack(torch.meshgrid(
            torch.arange(self.ratio, device=q_high_feat.device),
            torch.arange(self.ratio, device=q_high_feat.device),
            indexing='ij'), dim=-1).view(-1, 2)

        topleft = torch.stack([topk_indices_high_topleft_row, topk_indices_high_topleft_col], dim=-1)
        sparse_indices_2d = topleft.unsqueeze(-2) + delta.view(1, 1, 1, -1, 2)

        sparse_indices_1d = sparse_indices_2d[..., 0] * W + sparse_indices_2d[..., 1]
        sparse_indices_1d = sparse_indices_1d.view(B, N_low_q, -1)

        final_sparse_indices = sparse_indices_1d.unsqueeze(2).repeat(1, 1, self.ratio ** 2, 1).view(B, N_high_q, -1)
        indices_expanded = final_sparse_indices.unsqueeze(-1).expand(-1, -1, -1, C)

        # 先对完整的 K, V 进行投射
        k_high_proj = self.fine_attn_weights.k_proj(k_high_seq) # O(B * N * C^2)
        v_high_proj = self.fine_attn_weights.v_proj(v_high_seq) # O(B * N * C^2)
        
        # 再用同样的索引去 gather 已经投射过的数据
        # 注意：gather的输入源 k_high_proj 需要被正确地 expand
        indices_expanded_for_gather = final_sparse_indices.unsqueeze(-1).expand(-1, -1, -1, C)
        k_sparse_proj = torch.gather(k_high_proj.unsqueeze(1).expand(-1, N, -1, -1), 2, indices_expanded_for_gather) # O(B*N*k*R^2*C)
        v_sparse_proj = torch.gather(v_high_proj.unsqueeze(1).expand(-1, N, -1, -1), 2, indices_expanded_for_gather) # O(B*N*k*R^2*C)
        
        # 重命名变量以用于后续注意力计算
        k_proj = k_sparse_proj
        v_proj = v_sparse_proj
        # Fine-grained per-query attention
        head_dim = self.dim // self.num_heads
        K_sparse_len = final_sparse_indices.shape[-1]

        q_proj = self.fine_attn_weights.q_proj(q_high_seq)

        q = q_proj.view(B, N_high_q, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k_proj.view(B, N_high_q, K_sparse_len, self.num_heads, head_dim).permute(0, 3, 1, 2, 4)
        v = v_proj.view(B, N_high_q, K_sparse_len, self.num_heads, head_dim).permute(0, 3, 1, 2, 4)

        attn_scores = torch.einsum('bhnd,bhnkd->bhnk', q, k) * self.fine_attn_weights.scale
        attn_map = attn_scores.softmax(dim=-1)
        attn_map = self.fine_attn_weights.attn_drop(attn_map)

        x = torch.einsum('bhnk,bhnkd->bhnd', attn_map, v)

        x = x.transpose(1, 2).reshape(B, N_high_q, self.dim)
        warped_seq = self.fine_attn_weights.proj(x)
        warped_seq = self.fine_attn_weights.proj_drop(warped_seq)

        warped_feat = warped_seq.transpose(1, 2).view(B, C, H, W)
        return warped_feat


class SwinUnifiedAttention(nn.Module):
    """统一的Swin注意力核心 (窗口化)，包含相对位置编码。"""
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q * self.scale) @ k.transpose(-2, -1)
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================
# Part 3: 统一化的注意力Block模块
# ============================================================================

class UnifiedTransformerBlock(nn.Module):
    """统一的标准 Transformer Block。"""
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=2.0, qkv_bias=True,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn = StandardUnifiedAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        H, W = to_2tuple(input_resolution)
        dim_spatial = H * W

        self.q_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim))
        self.k_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim))
        
        self.norm_ffn = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), act_layer(), nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim), nn.Dropout(drop)
        )

    def forward(self, query, key=None, value=None):
        B, C, H, W = query.shape
        is_cross_attention = key is not None
        if not is_cross_attention:
            key, value = query, query

        q_in = query.flatten(2).transpose(1, 2)
        k_in = key.flatten(2).transpose(1, 2)
        v_in = value.flatten(2).transpose(1, 2)
        shortcut = v_in
        
        q_norm = self.norm_q(q_in + self.q_pos_embedding)
        k_norm = self.norm_kv(k_in + self.k_pos_embedding)
        v_norm = self.norm_kv(v_in)
        
        attn_output, _ = self.attn(query=q_norm, key=k_norm, value=v_norm)
        
        x = shortcut + attn_output
        x = x + self.mlp(self.norm_ffn(x))
        return x.transpose(1, 2).view(B, C, H, W)


class UnifiedSwinBlock(nn.Module):
    """统一的 Swin Transformer Block。"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn = SwinUnifiedAttention(
            dim, num_heads, self.window_size, qkv_bias, attn_drop, drop)
        
        self.norm_ffn = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), act_layer(),
            nn.Linear(mlp_hidden_dim, dim), nn.Dropout(drop))
        
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask.view(1,H,W,1), self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, query, key=None, value=None):
        B, C, H, W = query.shape
        is_cross_attention = key is not None
        if not is_cross_attention:
            key, value = query, query

        q = query.flatten(2).transpose(1, 2)
        k = key.flatten(2).transpose(1, 2)
        v = value.flatten(2).transpose(1, 2)
        shortcut = v
        
        q = self.norm_q(q).view(B, H, W, C)
        k = self.norm_kv(k).view(B, H, W, C)
        v = self.norm_kv(v).view(B, H, W, C)
        
        if self.shift_size > 0:
            shifted_q, shifted_k, shifted_v = [torch.roll(t, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                                               for t in (q, k, v)]
        else:
            shifted_q, shifted_k, shifted_v = q, k, v
        
        q_win = window_partition(shifted_q, self.window_size)
        k_win = window_partition(shifted_k, self.window_size)
        v_win = window_partition(shifted_v, self.window_size)
        
        attn_windows = self.attn(q_win, k_win, v_win, mask=self.attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm_ffn(x))
        return x.transpose(1, 2).view(B, C, H, W)


# ============================================================================
# Part 4: 统一的注意力层工厂
# ============================================================================

class AttentionLayerFactory(nn.Module):
    """一个统一的工厂模块，用于创建完整的（自）注意力层。"""
    def __init__(self, args, dim, resolution):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        common_kwargs = {
            'dim': dim,
            'input_resolution': resolution,
            'num_heads': args.num_heads,
        }

        if resolution[0] >= args.swin_res_threshold:
            print(f"  -> [工厂] 创建 SWIN 注意力层 @ {resolution}")
            self.blocks.append(UnifiedSwinBlock(window_size=args.window_size, shift_size=0, **common_kwargs))
            self.blocks.append(
                UnifiedSwinBlock(window_size=args.window_size, shift_size=args.window_size // 2, **common_kwargs))
        else:
            print(f"  -> [工厂] 创建 STANDARD 注意力层 @ {resolution}")
            self.blocks.append(UnifiedTransformerBlock(mlp_ratio=2.0, **common_kwargs))

    def forward(self, query, key=None, value=None):
        is_cross_attention = key is not None
        if is_cross_attention:
            v_out = value
            for block in self.blocks:
                v_out = block(query, key, v_out)
            return v_out
        else:
            x_out = query
            for block in self.blocks:
                x_out = block(x_out)
            return x_out


class CrossAttentionLayerFactory(nn.Module):
    """用于交叉注意力的工厂，创建粗粒度或精细粒度对齐模块。"""
    def __init__(self, args, dim, resolution):
        super().__init__()
        self.is_standard_attention = resolution[0] < args.swin_res_threshold

        if self.is_standard_attention:
            print(f"  -> [工厂] 创建 STANDARD 交叉注意力层 @ {resolution}")
            self.block = StandardUnifiedAttention(dim=dim, num_heads=args.num_heads)
        else:
            print(f"  -> [工厂] 创建 GUIDED SPARCE 交叉注意力层 @ {resolution}")
            anchor_resolution = args.swin_res_threshold
            ratio = 2 * (resolution[0] / anchor_resolution)
            assert ratio >= 1 and ratio.is_integer(), "Fine resolution must be a multiple of anchor resolution"
            self.block = GuidedResampler(dim=dim, downsample_ratio=int(ratio))

    def forward(self, query, key, value, attn=None):
        if not self.is_standard_attention:
            # 精细层接收 (B, C, H, W)
            x_out = self.block(value, attn.mean(dim=1))
            return x_out # 统一返回二元组
        else:
            # 粗糙层接收 (B, C, H, W) 并手动转换为序列
            B, C, H, W = query.shape
            query_seq = query.flatten(2).transpose(1, 2)
            key_seq = key.flatten(2).transpose(1, 2)
            value_seq = value.flatten(2).transpose(1, 2)
            
            x_out_seq, attn_map = self.block(query_seq, key_seq, value_seq)
            
            # 将输出转换回 (B, C, H, W)
            x_out = x_out_seq.transpose(1, 2).view(B, C, H, W)
            return x_out, attn_map