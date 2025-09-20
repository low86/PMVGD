import torch
from models.layers.hgc import *
from torch_geometric.nn import Linear
from torch_geometric.nn import global_mean_pool

def get_bounds_from_slice_dict(batch_obj):
    if not hasattr(batch_obj, '_slice_dict'):
        raise RuntimeError("The batch object does not have _slice_dict attribute.")
    key = 'visit'  
    slices = batch_obj._slice_dict[key]
    return slices['x']


def get_last_visit_features_from_slices(x, slices_tensor):
    last_visit_features = []
    for idx in range(slices_tensor.size(0) - 1):
        start, end = int(slices_tensor[idx]), int(slices_tensor[idx + 1])
        last_visit_feature = x[end - 1]
        last_visit_features.append(last_visit_feature)
    return torch.stack(last_visit_features)


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata): #图的元数据
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]: #将不同类型的节点特征（诊断、药物、就诊）映射到统一维度
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, edge_index_dict, batch):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in batch.x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, batch.edge_time_dict)

        visit_emb = global_mean_pool(x_dict['visit'], batch.batch_dict['visit'])
        co_emb = global_mean_pool(x_dict['co'], batch.batch_dict['co'])
        patient_emb = torch.cat([visit_emb, co_emb], dim=1)

        s = get_bounds_from_slice_dict(batch) #获得切片
        tmp  = get_last_visit_features_from_slices(x_dict['visit'],s)

        return self.lin(tmp), patient_emb
    
