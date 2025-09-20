import torch
import numpy as np
import torch_sparse
import torch.nn as nn
from typing import Optional, List
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, diags
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        
        time_dim = expand_dim // 2  
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        seq_len = ts.size(0)
        ts = ts.view(seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, -1)  
        map_ts += self.phase.view(1, -1)

        # Creating the time encoding with both cosine and sine parts
        harmonic = torch.cat([torch.cos(map_ts), torch.sin(map_ts)], dim=-1) 

        return harmonic
    
class Time2Vec(nn.Module):
    def __init__(self, output_size):
        super(Time2Vec, self).__init__()
        self.output_size = output_size//2
        self.linear = nn.Linear(1, self.output_size)  
        self.periodic = nn.Linear(1, self.output_size)  

    def forward(self, x):
        x = x.view(-1, 1)  
        linear_out = self.linear(x)
        periodic_out = torch.sin(self.periodic(x))
        return torch.cat([linear_out, periodic_out], dim=-1).view(-1, self.output_size * 2)
    


