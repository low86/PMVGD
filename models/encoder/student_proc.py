"""
Author: Tian Yuxuan
Date: 2025-08-16
"""
import torch
import torch.nn as nn
from models.layers.hgt import *
from models.bottleneck import *
from torch_geometric.nn import Linear
from models.layers.time_encoding import *
from utils.metapath_config import view_metas
from models.encoder.teacher import feats_to_nodes

class Student_proc(nn.Module):
    def __init__(self, Tokenizers, hidden_dim, output_size, device, num_heads=2, num_layers=1):
        super(Student_proc, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.embedding_dim = hidden_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.tim2vec = Time2Vec(8).to(device)
        self.feature_keys = Tokenizers.keys()
        self.metadata = view_metas['procedures_disease']

        self.lin_dict = torch.nn.ModuleDict()
        self.graphmodel = HGT(hidden_channels=hidden_dim, out_channels=output_size, num_heads=num_heads,
                              num_layers=num_layers, metadata=self.metadata).to(device)
        for node_type in self.metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)
        self.lin = Linear(hidden_dim, output_size)
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
        self.fc = Linear(hidden_dim * 3, output_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def add_feature_transform_layer(self, feature_key: str):  # 特征嵌入层创建方法
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),  # 词汇表大小
            self.embedding_dim,  # 嵌入维度
            padding_idx=tokenizer.get_padding_index(),  # 填充索引
        )

    def get_embedder(self):
        feature = {}
        for k in self.embeddings.keys():
            lenth = self.feat_tokenizers[k].get_vocabulary_size()
            tensor = torch.arange(0, lenth, dtype=torch.long).to(self.device)
            feature[k] = self.embeddings[k](tensor)
        return feature

    def process_graph_fea(self, graph_list, pe):
        f = self.get_embedder()
        for i in range(len(graph_list)):
            for node_type, x in graph_list[i].x_dict.items():
                if node_type != 'visit':
                    graph_list[i][node_type].x = f[feats_to_nodes[node_type]]
                else:
                    # 时间特征编码
                    timevec = self.tim2vec(
                        torch.tensor(graph_list[i]['visit'].time, dtype=torch.float32, device=self.device))
                    # 拼接患者表示和时间特征
                    num_visit = graph_list[i]['visit'].x.shape[0]
                    graph_list[i]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec], dim=-1)  # 患者表示加时序特征
        return graph_list  # 转换为批量图

    def forward(self, batch, pe):
        graph_data = self.process_graph_fea(batch, pe)
        graph_out, patient_emb = self.graphmodel(graph_data.edge_index_dict, graph_data)
        # x_dict = {
        #     node_type: self.lin_dict[node_type](x).relu_()
        #     for node_type, x in graph_data.x_dict.items()  # batch.x_dict 包含不同类型节点的原始特征
        # }
        # for layer in self.layers:
        #     x_dict = layer(x_dict, edge_index_dict, graph_data.edge_time_dict)

        # s = get_bounds_from_slice_dict(batch)
        # tmp = get_last_visit_features_from_slices(x_dict['visit'], s)
        pred = graph_out * self.alpha + self.fc(pe) * (1 - self.alpha)
        return pred, patient_emb


