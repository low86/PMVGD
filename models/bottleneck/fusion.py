"""
Author: Tian Yuxuan
Date: 2025-08-16
"""
import torch
import torch.nn as nn
from models.bottleneck import *
from .sparse_attention import TransformerBlock

class BottleneckFusion(nn.Module):
    def __init__(self, dim, num_tokens=3, num_heads=2, num_layers=1, dropout=0.2, num_modalities=3):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.bottlenecks = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.xavier_normal_(self.bottlenecks)
        self.type_embeddings = nn.Embedding(num_modalities, dim)
        self.cross_attn = TransformerBlock(embed_dim=dim, heads=num_heads, mlp_ratio=4.0, top_k=0.7, dropout=dropout, num_layer=num_layers)
        self.pos_embedding = nn.Parameter(torch.empty(1, 256, dim)) 
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, modality_list):
        B = modality_list[0].size(0)
        device = modality_list[0].device
        typed_inputs = []
        for mod_id, x in enumerate(modality_list):
            type_emb = self.type_embeddings(
                torch.tensor(mod_id, device=device, dtype=torch.long)
            ).view(1, 1, -1).expand(B, x.size(1), -1)
            typed_inputs.append(x + type_emb)
        all_modal = torch.cat(typed_inputs, dim=1)  # [B, sum(L_i), D]
        bottlenecks = self.bottlenecks.expand(B, -1, -1)
        combined = torch.cat([bottlenecks, all_modal], dim=1)  # [B, K+sum(L_i), D]
        L = combined.size(1)
        pos_emb = self.pos_embedding[:, :L, :]
        output = self.cross_attn(combined + pos_emb)
        return output[:, :self.num_tokens, :]

class Bottleneck_view(nn.Module):
    def __init__(self, dim, num_tokens=2, num_heads=2, num_layers=1, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.bottlenecks = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.xavier_normal_(self.bottlenecks)
        self.trans_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=dim,
                heads=num_heads,
                mlp_ratio=2.0,
                top_k=0.5,
                dropout=dropout,
                num_layer=1
            ) for _ in range(num_layers)
        ])

        self.pos_embedding = nn.Parameter(torch.empty(1, 32, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, visit_emb, co_emb):
        B, D = visit_emb.size(0), self.dim
        x_dual = torch.stack([visit_emb, co_emb], dim=1)  # [B, 2, D]
        bottlenecks = self.bottlenecks.expand(B, -1, -1)  # [B, K, D]
        combined = torch.cat([bottlenecks, x_dual], dim=1)  # [B, K+2, D]
        L = combined.size(1)
        pos_emb = self.pos_embedding[:, :L, :]
        for block in self.trans_blocks:
            combined = block(combined + pos_emb)
        return combined[:, :self.num_tokens, :]  # [B, K, D]