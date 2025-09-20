"""
Author: Tian Yuxuan
Date: 2025-08-16
"""
import torch
import torch.nn.functional as F

class ContrastiveLearner:
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def info_nce_loss(self, query, key):
        """
        query: (batch_size, dim) - e.g., from aux_model
        key: (batch_size, dim) - from main_model (anchor)
        """
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        logits = torch.matmul(query, key.T) / self.temperature  # shape (B, B)
        labels = torch.arange(query.size(0)).to(query.device)   # 正样本是对角线

        loss = F.cross_entropy(logits, labels)
        return loss
