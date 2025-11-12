"""
Date: 2025-08-16
"""
import torch
import torch.nn as  nn
from models.layers.hgt import *
from models.bottleneck import *
from models.layers.seq_models import *
from torch_geometric.data import Batch
from models.layers.time_encoding import *
from utils.metapath_config import view_metas

feats_to_nodes = {
    'cond_hist': 'co',
    'procedures': 'pr',
    'drugs': 'dh',
    'co': 'cond_hist',
    'pr': 'procedures',
    'dh': 'drugs'
}

class Teacher(nn.Module):
    def __init__(
            self,
            Tokenizers,
            hidden_size,
            output_size,
            device,
            graph_meta,
            dropout=0.5,
            num_heads=2,
            num_layers=2,
    ):
        super(Teacher, self).__init__()
        self.embedding_dim = hidden_size
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        self.bottleneck_fusion = BottleneckFusion(
            dim=hidden_size,
            num_tokens=3,
            num_heads=2,
            num_layers=1
        )

        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=hidden_size, dropout=dropout
            )
        self.tim2vec = Time2Vec(8).to(device)
        self.fc = nn.Linear(self.embedding_dim * (len(self.feature_keys)), output_size)
        self.graphmodel = HGT(hidden_channels=hidden_size, out_channels=output_size, num_heads=num_heads,
                              num_layers=num_layers, metadata=graph_meta).to(device)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.gammas = nn.Parameter(torch.zeros(len(self.feature_keys)))

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

    def get_embedder(self):
        feature = {}
        for k in self.embeddings.keys():
            lenth = self.feat_tokenizers[k].get_vocabulary_size()
            tensor = torch.arange(0, lenth, dtype=torch.long).to(self.device)
            feature[k] = self.embeddings[k](tensor)
        return feature

    def process_seq(self, seqdata):
        patient_emb = []
        patient_cls = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(seqdata[feature_key])
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)  # [B, T, F, D]
            x = torch.sum(x, dim=2)  # [B, T, D]
            mask = torch.any(x != 0, dim=2)
            x_encoded, x_cls = self.transformer[feature_key](x, mask)  # cls_emb: [B, D]
            patient_emb.append(x_encoded)
            patient_cls.append(x_cls)

        fused = self.bottleneck_fusion(patient_emb)  # [B, K=M, D]
        modality_reprs = [
            cls + torch.sigmoid(self.gammas[i]) * (fused[:, i, :] - cls)
            for i, cls in enumerate(patient_cls)
        ]
        fused_repr = torch.cat(modality_reprs, dim=1)  # [B, M*D]
        logits = self.fc(fused_repr)
        return logits, fused_repr

    def process_graph_fea(self, graph_list, pe):
        f = self.get_embedder()
        for i in range(len(graph_list)):
            for node_type, x in graph_list[i].x_dict.items():
                if node_type != 'visit':
                    graph_list[i][node_type].x = f[feats_to_nodes[node_type]]
                if node_type == 'visit':
                    timevec = self.tim2vec(
                        torch.tensor(graph_list[i]['visit'].time, dtype=torch.float32, device=self.device))
                    num_visit = graph_list[i]['visit'].x.shape[0]
                    graph_list[i]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec], dim=-1)
        return Batch.from_data_list(graph_list)

    def forward(self, batchdata):

        seq_logits, Patient_emb = self.process_seq(batchdata[0])

        graph_data = self.process_graph_fea(batchdata[1], Patient_emb).to(self.device)
        alpha = torch.clamp(self.alpha, 0, 1)
        graph_out, graph_emb = self.graphmodel(graph_data.edge_index_dict, graph_data)
        out = alpha * graph_out + (1 - alpha) * seq_logits
        return out, Patient_emb, graph_emb
