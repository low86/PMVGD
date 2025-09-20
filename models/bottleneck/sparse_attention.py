"""
Author: Tian Yuxuan
Date: 2025-08-16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, heads=2, top_k=0.7, dropout=0.2):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        if isinstance(top_k, float):
            assert 0 < top_k <= 1, "top_k ratio must be in (0, 1]"
            self.top_k_ratio = top_k
            self.top_k_fixed = None
        else:
            assert top_k >= 1, "top_k must be at least 1"
            self.top_k_fixed = top_k
            self.top_k_ratio = None

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, T, H * D)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(self._split_heads, (q, k, v))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if mask is not None:
            mask = mask.view(B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(~mask, -1e5)

        if self.top_k_ratio is not None:
            valid_len = mask.sum(dim=-1).max() if mask is not None else T
            top_k = max(1, int(self.top_k_ratio * valid_len))
        else:
            top_k = max(1, min(self.top_k_fixed, T))

        topk_vals, topk_idx = torch.topk(attn_scores, k=top_k, dim=-1)
        sparse_scores = torch.full_like(attn_scores, -1e4)
        sparse_scores.scatter_(-1, topk_idx, topk_vals)

        attn_weights = F.softmax(sparse_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = self._merge_heads(context)
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads=2, mlp_ratio=4.0, top_k=0.7, dropout=0.2, num_layer = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, heads, top_k, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer = num_layer
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Attention sublayer
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        # FFN sublayer
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x
