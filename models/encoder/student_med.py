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
from torch_geometric.data import Batch


class Student_med(nn.Module):
    def __init__(self, Tokenizers, hidden_dim, output_size, device, num_heads=2, num_layers=1):
        super(Student_med, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.embedding_dim = hidden_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.tim2vec = Time2Vec(8).to(device)
        self.feature_keys = Tokenizers.keys()
        self.metadata = self.metadata = view_metas['drug_disease']

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in self.metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)

        self.lin = Linear(hidden_dim, output_size)
        self.fc = Linear(hidden_dim*3, output_size)
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
        self.graphmodel = HGT(hidden_channels=hidden_dim, out_channels=output_size, num_heads=num_heads,
                              num_layers=num_layers, metadata=self.metadata).to(device)

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

    def process_graph_fea(self, graph_list, pe):
        graphs = graph_list.to_data_list() if isinstance(graph_list, Batch) else graph_list
        for i in range(len(graphs)):
            for node_type in graphs[i].node_types:
                if node_type != 'visit' and node_type in feats_to_nodes:
                    store = graphs[i][node_type]
                    node_id = getattr(store, 'node_id', None)
                    num_nodes = getattr(store, 'num_nodes', None)
                    if num_nodes is None:
                        if node_id is not None:
                            num_nodes = int(node_id.shape[0])
                        elif hasattr(store, 'x'):
                            num_nodes = int(store.x.shape[0])
                        else:
                            raise ValueError(f"Cannot infer num_nodes for node type: {node_type}")
                    if node_id is None:
                        node_id = torch.arange(int(num_nodes), dtype=torch.long)
                    node_id = node_id.to(self.device)
                    emb_key = feats_to_nodes[node_type]
                    graphs[i][node_type].x = self.embeddings[emb_key](node_id)
                if node_type == 'visit':
                    timevec = self.tim2vec(
                        torch.tensor(graphs[i]['visit'].time, dtype=torch.float32, device=self.device))
                    num_visit = int(getattr(graphs[i]['visit'], 'num_nodes', len(graphs[i]['visit'].time)))
                    graphs[i]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec], dim=-1)
        return Batch.from_data_list(graphs).to(self.device)

    def forward(self, batch, pe):
        graph_data = self.process_graph_fea(batch, pe)
        graph_out, patient_emb = self.graphmodel(graph_data.edge_index_dict, graph_data)
        # x_dict = {
        #     node_type: self.lin_dict[node_type](x).relu_()
        #     for node_type, x in graph_data.x_dict.items()  # batch.x_dict 包含不同类型节点的原始特征
        # }
        # for layer in self.layers:
        #     x_dict = layer(x_dict, edge_index_dict, graph_data.edge_time_dict)
        #
        # visit_emb = global_mean_pool(x_dict['visit'], graph_data.batch_dict['visit'])
        # co_emb = global_mean_pool(x_dict['co'], graph_data.batch_dict['co'])
        #
        # patient_emb = torch.cat([visit_emb, co_emb], dim=1)
        # s = get_bounds_from_slice_dict(batch)
        # tmp = get_last_visit_features_from_slices(x_dict['visit'], s)
        pred = graph_out * self.alpha + self.fc(pe) * (1 - self.alpha)
        return pred, patient_emb

