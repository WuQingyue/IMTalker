import os, math, torch
from weakref import ref
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from models import BaseModel

from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------- Helper Functions (修改后) -----------------

def enc_dec_mask(T, S, device, frame_width = 1, expansion = 2):
    """
    生成一个局域注意力掩码 (Local Attention Mask).
    True 代表需要被 mask (忽略).
    """
    mask = torch.ones(T, S, device=device, dtype=torch.bool)
    for i in range(T):
        # 将注意力窗口内的值设为 False (不 mask)
        start = max(0, (i - expansion) * frame_width)
        end = min(S, (i + expansion + 1) * frame_width)
        mask[i, start:end] = False
    return mask


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Sinusoidal position encoding table.
    """
    def cal_angle(position, hid_idx):
        return position / (10000 ** (2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = torch.Tensor([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return sinusoid_table


# ----------------- Core Modules (修改后) -----------------

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # 检查 PyTorch 版本是否支持 Fused Attention
        self.fused_attn = hasattr(F, "scaled_dot_product_attention")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # PyTorch 2.0+ Fused Attention
        if self.fused_attn:
            # mask 为 True 的位置会被忽略
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=~mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            
        # 手动实现 Attention
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                # 在 softmax 之前应用 mask
                # mask 中为 True 的位置填充为 -inf
                attn = attn.masked_fill(mask, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class SequenceEmbed(nn.Module):
    def __init__(self, dim_w, dim_h, norm_layer=None, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_w, dim_h, bias=bias)
        self.norm = norm_layer(dim_h) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))

class FMTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # Mlp 在你的代码中未提供，这里假设它是一个标准的 MLP 模块
        # 如果 Mlp 未定义，请替换为下面的 nn.Sequential
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_size, mlp_hidden_dim),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(mlp_hidden_dim, hidden_size),
        #     nn.Dropout(0)
        # )
        from timm.models.layers import Mlp # 假设 Mlp 来自 timm 库
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x, c, mask=None) -> torch.Tensor:
        # mask is expected to be of shape [B, num_heads, T, T] or broadcastable
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(self.framewise_modulate(self.norm1(x), shift_msa, scale_msa), mask=mask)
        x = x + gate_mlp * self.mlp(self.framewise_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_size, dim_w):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, dim_w, bias=True)

    def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.framewise_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

# ----------------- Main Model (修改后) -----------------
class FlowMatchingTransformer(nn.Module): # 建议从 nn.Module 继承
    def __init__(self, opt, max_seq_len=2048) -> None: # 增加 max_seq_len
        super().__init__()
        self.opt = opt
        self.max_seq_len = max_seq_len # 用于位置编码
        
        self.hidden_size = opt.dim_h
        self.mlp_ratio = opt.mlp_ratio
        self.fmt_depth = opt.fmt_depth
        self.num_heads = opt.num_heads
        self.attention_window = opt.attention_window # 保存 attention window

        self.x_embedder = SequenceEmbed(opt.dim_motion + opt.dim_motion, self.hidden_size) # 假设 ref_x 的维度也是 opt.dim_motion
        
        # 动态位置编码：创建一个足够长的 buffer，然后在使用时切片
        pos_embed_table = get_sinusoid_encoding_table(self.max_seq_len, self.hidden_size)
        self.register_buffer('pos_embed', pos_embed_table.unsqueeze(0))

        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.c_embedder = nn.Linear(opt.dim_c, self.hidden_size) # 假设 a 的维度是 opt.dim_c

        self.blocks = nn.ModuleList([
            FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.fmt_depth)
        ])
        self.decoder = Decoder(self.hidden_size, self.opt.dim_motion)
        
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.linear.weight, 0)
        nn.init.constant_(self.decoder.linear.bias, 0)

    def sequence_embedder(self, sequence, dropout_prob, train=False) -> torch.Tensor:
        if train and dropout_prob > 0:
            batch_id_for_drop = torch.where(torch.rand(sequence.shape[0], device=sequence.device) < dropout_prob)
            sequence[batch_id_for_drop] = 0
        return sequence

    # ---------------- vvv 主要修改区域 vvv ------------------
    def forward(self, batch, t, train=True) -> torch.Tensor:
        # 1. 从新的 batch 格式中解包数据
        m_full = batch["m_full"]
        a_full = batch["a_full"]
        m_ref = batch["m_ref"]
        split_points = batch["split_points"]
        m_full_lens = batch["m_full_lens"]
        
        B, T, _ = m_full.shape
    
        # 创建一个可复用的范围张量
        arange = torch.arange(T, device=m_full.device).unsqueeze(0)
    
        # --- 2. 准备 Motion 输入 (x) 并应用条件丢弃 ---
        prev_mask = (arange < split_points.unsqueeze(1)).unsqueeze(-1)
        prob_m = 0.5 if train else 0.0
        batch_drop_decision_m = torch.rand(B, 1, 1, device=m_full.device) < prob_m
        final_dropout_mask_m = prev_mask & batch_drop_decision_m
        x = m_full.masked_fill(final_dropout_mask_m, 0)
        
        # --- 3. 准备 Reference 输入 (ref_x)，不进行丢弃 ---
        ref_x_expanded = m_ref[:, None, :].repeat(1, T, 1)
    
        # --- 4. 组合 Motion 和 Reference，并送入 Embedder ---
        x_with_ref = torch.cat([ref_x_expanded, x], dim=-1)
        x_embedded = self.x_embedder(x_with_ref)
        
        # --- 5. 添加位置编码 ---
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        x_final = x_embedded + self.pos_embed[:, :T, :]
        
        # --- 6. 准备 Audio 条件 (c) 并应用条件丢弃 ---
        prob_a = 0.5 if train else 0.0
        batch_drop_decision_a = torch.rand(B, 1, 1, device=a_full.device) < prob_a
        final_dropout_mask_a = prev_mask & batch_drop_decision_a
        a_full_dropped = a_full.masked_fill(final_dropout_mask_a, 0)
        
        # --- 7. 准备最终的条件向量 c ---
        c = self.c_embedder(a_full_dropped) 
        t_emb = self.t_embedder(t).unsqueeze(1)
        c = c + t_emb
    
        # --- 8. 创建注意力掩码 (*** 仅处理 Padding ***) ---
        # a. 创建 Padding Mask，True 代表需要被忽略的位置
        padding_mask = arange >= m_full_lens.unsqueeze(1) # Shape: [B, T]
        
        # b. 扩展到多头注意力的维度 [B, H, T, T]
        #    这个掩码的含义是：对于批次b, 如果key j是padding, 则所有query i都不能关注它
        attn_mask = padding_mask.view(B, 1, 1, T).expand(-1, self.num_heads, T, -1)
    
        # --- 9. 通过 Transformer 模块进行前向传播 ---
        for block in self.blocks:
            x_final = block(x_final, c, mask=attn_mask)
        
        # --- 10. 解码器输出最终结果 ---
        return self.decoder(x_final, c)
        
	
@torch.no_grad()
def forward_with_cfv(self, t, x, a,  prev_x, prev_a, ref_x, a_cfg_scale=1.0, **kwargs) -> torch.Tensor:#
		if a_cfg_scale != 1.0 :
			null_a = torch.zeros_like(a)

			audio_cat 	= torch.cat([null_a, a], dim=0)
			#pose_cat    = torch.cat([pose, pose], dim=0)
			#gaze_cat    = torch.cat([gaze, gaze], dim=0)
			x 			= torch.cat([x, x], dim=0)					# concat along batch

			prev_x_cat  = torch.cat([prev_x, prev_x], dim=0)
			prev_a_cat = torch.cat([prev_a, prev_a], dim=0)
			#prev_pose_cat = torch.cat([prev_pose,prev_pose], dim=0)
			#prev_gaze_cat = torch.cat([prev_gaze, prev_gaze], dim=0)
			ref_x      = torch.cat([ref_x, ref_x], dim=0)	
			model_output = self.forward(t, x, audio_cat, prev_x_cat, prev_a_cat, ref_x, train=False)
			#model_output = self.forward(t, x, audio_cat, prev_x_cat, prev_a_cat, pose_cat, gaze_cat, prev_pose_cat, prev_gaze_cat, train=False)
			uncond, all_cond = torch.chunk(model_output, chunks=2, dim=0)
			# Classifier-free vector field 	(cfv) incremental manner
			return uncond + a_cfg_scale * (all_cond - uncond)
		else:
			return self.forward(t, x, a, prev_x, prev_a, ref_x, train = False)

