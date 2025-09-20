"""
Author: Tian Yuxuan
Date: 2025-08-04
"""
import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

class MMDataset(Dataset):
    def __init__(self,
                 id_list_file,
                 graph_pkl_path,
                 seq_pkl_path):

        with open(id_list_file, 'r') as f:
            self.id_list = [int(line.strip()) for line in f]

        with open(graph_pkl_path, 'rb') as f:
            self.multi_view_graphs = pickle.load(f)

        with open(seq_pkl_path, 'rb') as f:
            self.seq_data = pickle.load(f)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        patient_id = self.id_list[idx]
        graph = self.multi_view_graphs[patient_id]
        seq = self.seq_data[patient_id]

        return  graph, seq

def split_dataset(id_split_dir, graph_pkl_path, seq_pkl_path):

    trainset = MMDataset(
        id_list_file=os.path.join(id_split_dir, 'train_ids.txt'),
        graph_pkl_path=graph_pkl_path,
        seq_pkl_path=seq_pkl_path
    )
    validset = MMDataset(
        id_list_file=os.path.join(id_split_dir, 'val_ids.txt'),
        graph_pkl_path=graph_pkl_path,
        seq_pkl_path=seq_pkl_path
    )
    testset = MMDataset(
        id_list_file=os.path.join(id_split_dir, 'test_ids.txt'),
        graph_pkl_path=graph_pkl_path,
        seq_pkl_path=seq_pkl_path
    )
    return trainset, validset, testset

def custom_collate_fn(batch):
    graph_data_list = [item[0] for item in batch]
    sequence_data_list = [item[1] for item in batch]

    graph_data_batch = graph_data_list
    sequence_data_batch = {key: [d[key] for d in sequence_data_list] for key in sequence_data_list[0]}

    return  sequence_data_batch, graph_data_batch



def mm_dataloader(trainset, validset, testset, batch_size=128):
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class ViewDataset(Dataset):
    def __init__(self, id_list_file, graph_pkl_path, seq_pkl_path, drug_pkl_path, procedure_pkl_path):

        with open(id_list_file, 'r') as f:
            self.id_list = [int(line.strip()) for line in f]

        with open(graph_pkl_path, 'rb') as f:
            self.multi_view_graphs = pickle.load(f)

        with open(seq_pkl_path, 'rb') as f:
            self.seq_data = pickle.load(f)

        with open(drug_pkl_path, 'rb') as f:
            self.drug_data = pickle.load(f)

        with open(procedure_pkl_path, 'rb') as f:
            self.procedure_data = pickle.load(f)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        patient_id = self.id_list[idx]
        graph = self.multi_view_graphs[patient_id]
        seq = self.seq_data[patient_id]
        drug = self.drug_data[patient_id]
        procedure = self.procedure_data[patient_id]
        visit_seq = graph
        return  graph, seq, drug, procedure, visit_seq, patient_id


def prepare_aux_dataset(id_split_dir, graph_pkl_path, seq_pkl_path, drug_pkl_path, procedure_pkl_path):
    trainset = ViewDataset(
        id_list_file=os.path.join(id_split_dir, f"train_ids.txt"),
        graph_pkl_path=graph_pkl_path,
        seq_pkl_path=seq_pkl_path,
        drug_pkl_path=drug_pkl_path,
        procedure_pkl_path=procedure_pkl_path,
    )
    validset = ViewDataset(
        id_list_file=os.path.join(id_split_dir, f"val_ids.txt"),
        graph_pkl_path=graph_pkl_path,
        seq_pkl_path=seq_pkl_path,
        drug_pkl_path=drug_pkl_path,
        procedure_pkl_path=procedure_pkl_path,

    )
    testset = ViewDataset(
        id_list_file=os.path.join(id_split_dir, f"test_ids.txt"),
        graph_pkl_path=graph_pkl_path,
        seq_pkl_path=seq_pkl_path,
        drug_pkl_path=drug_pkl_path,
        procedure_pkl_path=procedure_pkl_path,

    )
    return trainset, validset, testset

def custom_collate_view(batch):
    graph_data_list = [item[0] for item in batch]
    sequence_data_list = [item[1] for item in batch]
    drug_data_list = [item[2] for item in batch]  # 新增
    procedure_data_list = [item[3] for item in batch]
    patient_id_list = [item[4] for item in batch]  # 新增

    graph_data_batch = graph_data_list
    sequence_data_batch = {key: [d[key] for d in sequence_data_list] for key in sequence_data_list[0]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return  (sequence_data_batch, graph_data_batch, Batch.from_data_list(drug_data_list).to(device),
             Batch.from_data_list(procedure_data_list).to(device), patient_id_list)



def view_dataloader(train_aux, val_set, test_set, batch_size=128):
    train_loader = DataLoader(train_aux, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_view)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_view)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_view)
    return train_loader, val_loader, test_loader

def split_dataset_chunk(id_split_dir, graph_pkl_path, seq_pkl_path, chunk):

    trainset = MMDataset(
        id_list_file=os.path.join(id_split_dir, f'train_ids_chunk{chunk}.txt'),
        graph_pkl_path=graph_pkl_path,
        seq_pkl_path=seq_pkl_path
    )
    validset = MMDataset(
        id_list_file=os.path.join(id_split_dir, f'val_ids_chunk{chunk}.txt'),
        graph_pkl_path=graph_pkl_path,
        seq_pkl_path=seq_pkl_path
    )
    testset = MMDataset(
        id_list_file=os.path.join(id_split_dir, f'test_ids_chunk{chunk}.txt'),
        graph_pkl_path=graph_pkl_path,
        seq_pkl_path=seq_pkl_path
    )
    return trainset, validset, testset

import os
import pickle
from torch.utils.data import IterableDataset, DataLoader

def read_ids(id_file):
    with open(id_file, 'r') as f:
        return [int(line.strip()) for line in f]

class LazyChunkDataset(IterableDataset):
    def __init__(self, id_file, graph_pkl_path, seq_pkl_path):
        self.id_list = read_ids(id_file)
        self.graph_pkl_path = graph_pkl_path
        self.seq_pkl_path = seq_pkl_path

    def __iter__(self):
        # 每次迭代时再加载数据，节省内存
        with open(self.graph_pkl_path, 'rb') as f:
            graph_data = pickle.load(f)
        with open(self.seq_pkl_path, 'rb') as f:
            seq_data = pickle.load(f)

        for pid in self.id_list:
            yield seq_data[pid], graph_data[pid]

def lazy_collate_fn(batch):
    seq_batch = [item[0] for item in batch]
    graph_batch = [item[1] for item in batch]
    seq_out = {key: [d[key] for d in seq_batch] for key in seq_batch[0]}
    return seq_out, graph_batch

def mm_dataloader_lazy(id_split_dir, graph_pkl_path, seq_pkl_path, batch_size=128):
    def make_loader(split):
        dataset = LazyChunkDataset(
            id_file=os.path.join(id_split_dir, f'{split}_ids.txt'),
            graph_pkl_path=graph_pkl_path,
            seq_pkl_path=seq_pkl_path
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=lazy_collate_fn
        )
    return make_loader('train'), make_loader('val'), make_loader('test')
