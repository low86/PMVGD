import torch
import math
import torch.nn as  nn
from typing import Optional, Tuple, List


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        if mask is not None:
            mask = mask.sum(dim=-1) > 0
            x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=4 * hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)


class TransformerLayer(nn.Module):
    def __init__(self, feature_size, heads=2, dropout=0.5, num_layers=1):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerBlock(feature_size, heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.tensor, mask: Optional[torch.tensor] = None
    ) -> Tuple[torch.tensor, torch.tensor]:

        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb