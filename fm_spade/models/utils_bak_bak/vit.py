
from pytz import NonExistentTimeError
import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_spatial, mlp_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # pos embeddings: shape (1, seq_len, dim)
        self.q_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim))
        self.k_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W

        # reshape to (seq_len, batch, dim)
        x_reshaped = x.view(B, C, seq_len).permute(2, 0, 1)
        x_norm = self.norm1(x_reshaped)

        # prepare position embeddings: expand to (seq_len, batch, dim)
        # q_pos_embedding: (1, seq_len, dim) -> (seq_len, batch, dim)
        q_pos = self.q_pos_embedding.expand(-1, -1, -1)         # (1, seq_len, dim)
        q_pos = q_pos.permute(1, 0, 2).expand(seq_len, B, C)    # -> (seq_len, batch, dim)

        k_pos = self.k_pos_embedding.expand(-1, -1, -1)
        k_pos = k_pos.permute(1, 0, 2).expand(seq_len, B, C)

        # apply attention with position-augmented queries/keys
        att_output, attn_weights = self.attention(x_norm + q_pos,
                                                  x_norm + k_pos,
                                                  x_norm)
        x_reshaped = x_reshaped + att_output

        # feed-forward
        ff_output = self.mlp(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + ff_output

        # reshape back to (B, C, H, W)
        output = x_reshaped.permute(1, 2, 0).view(B, C, H, W)
        return output



class CrossAttentionModule(nn.Module):
    def __init__(self, 
        dim_spatial=4096,
        dim_qk=256,
        dim_v=256
        ):
        super().__init__()

        self.dim_head = dim_qk
        self.scale = dim_qk ** -0.5

        #print("CrossAttentionModule:",dim_spatial)
        #print("dim_qk:",dim_qk)
        #print("dim_v:",dim_v)
        

        # Separate positional encodings for queries and keys
        self.q_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.k_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.attend = nn.Softmax(dim=-1)
    def forward(self, queries, keys, values):
        # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
        q = torch.flatten(queries, start_dim=2).transpose(-1, -2)
        q = q + self.q_pos_embedding  # (b, dim_spatial, dim_qk)

        # in paper, key dim_spatial may be different from query dim_spatial
        # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
        k = torch.flatten(keys, start_dim=2).transpose(-1, -2)
        k = k + self.k_pos_embedding  # (b, dim_spatial, dim_qk)
        # (b, dim_v, h, w) -> (b, dim_v, dim_spatial) -> (b, dim_spatial, dim_v)
        v = torch.flatten(values, start_dim=2).transpose(-1, -2)

        # # (b, dim_spatial, dim_qk) * (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_spatial)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)  # (b, dim_spatial, dim_spatial)

        # (b, dim_spatial, dim_spatial) * (b, dim_spatial, dim_v) -> (b, dim_spatial, dim_v)
        out = torch.matmul(attn, v)

        # Or the torch version fast attention
        # out = F.scaled_dot_product_attention(q, k, v)
        out = torch.reshape(out.transpose(-1, -2), values.shape)  # (b, dim_spatial, dim_v) -> (b, dim_v, h, w)

        return out

class CrossAttentionModule_bias(nn.Module):
    def __init__(self, 
        dim_spatial=4096,
        dim_qk=256,
        dim_v=256
        ):
        super().__init__()

        self.dim_head = dim_qk
        self.scale = dim_qk ** -0.5

        #print("CrossAttentionModule:",dim_spatial)
        #print("dim_qk:",dim_qk)
        #print("dim_v:",dim_v)
        

        # Separate positional encodings for queries and keys
        self.q_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.k_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.bias_mouth = nn.Parameter(torch.randn(1))
        self.bias_eye = nn.Parameter(torch.randn(1))
        self.bias = [self.bias_eye, self.bias_mouth]
        self.attend = nn.Softmax(dim=-1)
    def forward(self, queries, keys, values, mask_list=None):
        # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
        q = torch.flatten(queries, start_dim=2).transpose(-1, -2)
        q = q + self.q_pos_embedding  # (b, dim_spatial, dim_qk)

        # in paper, key dim_spatial may be different from query dim_spatial
        # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
        k = torch.flatten(keys, start_dim=2).transpose(-1, -2)
        k = k + self.k_pos_embedding  # (b, dim_spatial, dim_qk)
        # (b, dim_v, h, w) -> (b, dim_v, dim_spatial) -> (b, dim_spatial, dim_v)
        v = torch.flatten(values, start_dim=2).transpose(-1, -2)

        # # (b, dim_spatial, dim_qk) * (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_spatial)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)  # (b, dim_spatial, dim_spatial)
        if mask_list:
            for idx, mask in enumerate(mask_list):

                if mask is not None:
                    resized_mask = F.interpolate(mask, size=(queries.shape[-2], queries.shape[-1]), mode='nearest')
                    mask = torch.flatten(resized_mask, start_dim=2)
                    mask_map = torch.matmul(mask, mask.transpose(-1, -2))
                    bias = self.bias[idx] * mask_map
                    attn = attn + torch.nn.functional.softplus(bias)

        # (b, dim_spatial, dim_spatial) * (b, dim_spatial, dim_v) -> (b, dim_spatial, dim_v)
        out = torch.matmul(attn, v)

        # Or the torch version fast attention
        # out = F.scaled_dot_product_attention(q, k, v)
        out = torch.reshape(out.transpose(-1, -2), values.shape)  # (b, dim_spatial, dim_v) -> (b, dim_v, h, w)

        return out
    

class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, spatial_dim, depth=2, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = CrossAttentionModule(dim_spatial=spatial_dim[0] * spatial_dim[0], dim_qk=motion_dim, dim_v=feature_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, heads, spatial_dim[0] * spatial_dim[0], mlp_dim) for _ in range(depth)
        ])
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        
    def forward_cross_attention(self, q, k, v):
        return self.cross_attention(q, k, v) #(b, dim_v, h, w)

    def forward_transformer_blocks(self, V_prime):
        for block in self.transformer_blocks:
            V_prime = block(V_prime)
        return V_prime

    def forward(self, ml_c, ml_r, fl_r):
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        for i, block in enumerate(self.transformer_blocks):
            V_prime  = block(V_prime)
        return V_prime

class ImplicitMotionAlignment_bias(nn.Module):
    def __init__(self, feature_dim, motion_dim, spatial_dim, depth=2, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = CrossAttentionModule_bias(dim_spatial=spatial_dim[0] * spatial_dim[0], dim_qk=motion_dim, dim_v=feature_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, heads, spatial_dim[0] * spatial_dim[0], mlp_dim) for _ in range(depth)
        ])
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        
    def forward_cross_attention(self, q, k, v, mask_list):
        return self.cross_attention(q, k, v, mask_list) #(b, dim_v, h, w)


    def forward_transformer_blocks(self, V_prime):
        for block in self.transformer_blocks:
            V_prime = block(V_prime)
        return V_prime

    def forward(self, ml_c, ml_r, fl_r, mask_list):
        V_prime = self.cross_attention(ml_c, ml_r, fl_r, mask_list)
        for i, block in enumerate(self.transformer_blocks):
            V_prime  = block(V_prime)
        return V_prime
    
# Example usage
if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    # Example dimensions
    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    heads = 8
    dim_head = 64
    mlp_dim = 1024
    t = 1


    # Create random input tensors and move to device
    ml_c = torch.randn(B, C_m, H, W).to(device)
    ml_r = torch.randn(B, C_m, H, W).to(device)
    fl_r = torch.randn(B, C_f, H, W).to(device)
    mask_list = [torch.randn(B, 1, 256, 256).to(device)]

    # Initialize the ImplicitMotionAlignment module and move to device
    model = ImplicitMotionAlignment_bias(feature_dim, motion_dim, (H,W), heads, dim_head, mlp_dim).to(device)

    # Forward pass
    with torch.no_grad():
        output = model.forward_cross_attention(ml_c, ml_r, fl_r, mask_list)
        output = model.forward_transformer_blocks(output)