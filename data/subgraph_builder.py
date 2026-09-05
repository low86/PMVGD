"""
Author: Tian Yuxuan
Date: 2025-07-16
"""
import pickle
from utils.utils import *
from datetime import datetime
from itertools import product
from torch_geometric.data import HeteroData
from utils.metapath_config import view_metas


class GraphBuilder:
    view_config = {
        'patient_disease': ['visit', 'co'],
        'drug_disease': ['visit', 'dh', 'co'],
        'procedures_disease': ['visit', 'pr', 'co'],
    }

    def __init__(self, dataset, tokenizer, dim=128, device='gpu', cache_dir='graph_cache'):
        self.dataset = dataset  # list of dicts (each is a patient sample)
        self.c_tokenzier = tokenizer['cond_hist']
        self.d_tokenzier = tokenizer['drugs']
        self.p_tokenzier = tokenizer['procedures']
        self.dim = dim
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.seqdata = []  # only store for full graph (patient_disease)

    @staticmethod
    def _unique_tokens_and_map(token_lists, fallback_token):
        flat = [tok for sub in token_lists for tok in sub]
        unique_tokens = sorted(set(flat))
        if len(unique_tokens) == 0:
            unique_tokens = [int(fallback_token)]
        token_to_local = {tok: i for i, tok in enumerate(unique_tokens)}
        return unique_tokens, token_to_local

    def build_cooccurrence_edge_with_weight_fast(self, node_lists):
        all_edges = []
        for nodes in node_lists:
            if len(nodes) < 2:
                continue
            pairs = torch.combinations(torch.tensor(nodes), r=2)
            all_edges.append(pairs)
            all_edges.append(pairs.flip(1))
        if not all_edges:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)

        all_edges = torch.cat(all_edges, dim=0)
        edge_index, counts = torch.unique(all_edges, dim=0, return_counts=True)
        return edge_index.T, torch.log(1 + counts.float())

    def build_single_graph(self, dp_data, view_name):
        data = HeteroData()
        num_visit = len(dp_data['procedures'])
        included_nodes = self.view_config[view_name]
        if 'adm_time' in dp_data and len(dp_data['adm_time']) >= 2:
            date_strings = [t.split()[0] for t in dp_data['adm_time']]
            time_stamps = [
                datetime.strptime(date, "%Y-%m-%d").timestamp()
                for date in date_strings
            ]
            time_diffs_days = [
                (time_stamps[i + 1] - time_stamps[i]) / (3600 * 24)
                for i in range(len(time_stamps) - 1)
            ]
            weights = 1 / (1 + torch.tensor(time_diffs_days, dtype=torch.float32))
        else:
            weights = torch.ones(num_visit - 1, dtype=torch.float32)

        if 'visit' in included_nodes:
            data['visit'].num_nodes = num_visit
            data['visit'].time = convert_to_relative_time(dp_data['adm_time'])

        if 'co' in included_nodes:
            dpc = self.c_tokenzier.batch_encode_2d(dp_data['cond_hist'], padding=False)
            co_pad = self.c_tokenzier.get_padding_index()
            co_tokens, co_map = self._unique_tokens_and_map(dpc, co_pad)
            data['co'].node_id = torch.tensor(co_tokens, dtype=torch.long)
            data['co'].num_nodes = len(co_tokens)

            if view_name == 'disease_cooccurrence':
                edge_index, edge_weight = self.build_cooccurrence_edge_with_weight_fast(dpc)
                data['co', 'cooccur', 'co'].edge_index = edge_index
                data['co', 'cooccur', 'co'].edge_attr = edge_weight  # or edge_attr

            elif 'visit' in included_nodes:
                civ = torch.tensor([[co_map[item] for sublist in dpc for item in sublist],
                                    [index for index, sublist in enumerate(dpc) for _ in sublist]], dtype=torch.int64)
                data['co', 'in', 'visit'].edge_index = civ
                data['co', 'in', 'visit'].edge_time = torch.tensor([index for index, sublist in enumerate(dpc) for _ in sublist], dtype=torch.float32)
                data['visit', 'has', 'co'].edge_index = torch.flip(civ, [0])

        if 'pr' in included_nodes:
            dpp = self.p_tokenzier.batch_encode_2d(dp_data['procedures'], padding=False)
            pr_pad = self.p_tokenzier.get_padding_index()
            pr_tokens, pr_map = self._unique_tokens_and_map(dpp, pr_pad)
            data['pr'].node_id = torch.tensor(pr_tokens, dtype=torch.long)
            data['pr'].num_nodes = len(pr_tokens)
            piv = torch.tensor([[pr_map[item] for sublist in dpp for item in sublist],
                                [index for index, sublist in enumerate(dpp) for _ in sublist]], dtype=torch.int64)
            data['pr', 'in', 'visit'].edge_index = piv
            data['pr', 'in', 'visit'].edge_time = torch.tensor([index for index, sublist in enumerate(dpp) for _ in sublist], dtype=torch.float32)
            data['visit', 'has', 'pr'].edge_index = torch.flip(piv, [0])

        if 'dh' in included_nodes:
            dpd = self.d_tokenzier.batch_encode_2d(dp_data['drugs'], padding=False)
            dh_pad = self.d_tokenzier.get_padding_index()
            dh_tokens, dh_map = self._unique_tokens_and_map(dpd, dh_pad)
            data['dh'].node_id = torch.tensor(dh_tokens, dtype=torch.long)
            data['dh'].num_nodes = len(dh_tokens)

            if view_name == 'drug_cooccurrence':
                edge_index, edge_weight = self.build_cooccurrence_edge_with_weight_fast(dpd)
                data['dh', 'cooccur', 'dh'].edge_index = edge_index
                data['dh', 'cooccur', 'dh'].edge_attr = edge_weight

            elif 'visit' in included_nodes:
                div = torch.tensor([[dh_map[item] for sublist in dpd for item in sublist],
                                    [index for index, sublist in enumerate(dpd) for _ in sublist]], dtype=torch.int64)
                data['dh', 'in', 'visit'].edge_index = div
                data['dh', 'in', 'visit'].edge_time = torch.tensor([index for index, sublist in enumerate(dpd) for _ in sublist], dtype=torch.float32)
                data['visit', 'has', 'dh'].edge_index = torch.flip(div, [0])

        if view_name in ['patient_disease', 'treatment_path', 'visit_sequence', 'drug_disease', 'procedures_disease']:
            if num_visit > 1:
                viv = torch.tensor([[i for i in range(num_visit - 1)],
                                    [i + 1 for i in range(num_visit - 1)]], dtype=torch.int64)
                data['visit', 'connect', 'visit'].edge_index = viv
                data['visit', 'connect', 'visit'].edge_attr = weights

            if view_name == 'treatment_path' and 'pr' in included_nodes and 'dh' in included_nodes:
                dpp = self.p_tokenzier.batch_encode_2d(dp_data['procedures'], padding=False)
                dpd = self.d_tokenzier.batch_encode_2d(dp_data['drugs'], padding=False)

                pr_dh_edges = [list(pair) for procedures, drugs in zip(dpp, dpd) for pair in product(procedures, drugs)]

                if pr_dh_edges:
                    pr_dh_tensor = torch.tensor(pr_dh_edges, dtype=torch.int64).T
                    data['pr', 'to', 'dh'].edge_index = pr_dh_tensor

        self._ensure_view_schema(data, view_name)
        return data

    def _ensure_view_schema(self, data, view_name):
        """Populate empty stores so each graph matches its declared metadata."""
        node_types, edge_types = view_metas[view_name]
        for node_type in node_types:
            if node_type not in data.node_types:
                data[node_type].num_nodes = 0
            elif getattr(data[node_type], 'num_nodes', None) is None:
                node_id = getattr(data[node_type], 'node_id', None)
                data[node_type].num_nodes = int(node_id.shape[0]) if node_id is not None else 0

        for src, rel, dst in edge_types:
            edge_store = data[src, rel, dst]
            if 'edge_index' not in edge_store:
                edge_store.edge_index = torch.empty((2, 0), dtype=torch.long)
            if rel == 'in' and 'edge_time' not in edge_store:
                edge_store.edge_time = torch.empty((0,), dtype=torch.float32)
            if rel == 'connect' and 'edge_attr' not in edge_store:
                edge_store.edge_attr = torch.empty((0,), dtype=torch.float32)

    # def build_and_save_all(self, chunk_size=11000):
    #     num_patients = len(self.dataset)
    #     num_chunks = (num_patients + chunk_size - 1) // chunk_size
    #
    #     for view_name in self.view_config:
    #         print(f'Building graphs for view: {view_name}')
    #         for chunk_id in range(num_chunks):
    #             start_idx = chunk_id * chunk_size
    #             end_idx = min((chunk_id + 1) * chunk_size, num_patients)
    #             graph_list = []
    #             seqdata_chunk = []
    #
    #             for dp_data in tqdm(self.dataset[start_idx:end_idx], desc=f'{view_name} Chunk {chunk_id}'):
    #                 graph = self.build_single_graph(dp_data, view_name)
    #                 graph_list.append(graph)
    #
    #                 if view_name == 'patient_disease':
    #                     seqdata_chunk.append({
    #                         'adm_time': dp_data['adm_time'],
    #                         'procedures': dp_data['procedures'],
    #                         'drugs': dp_data['drugs'],
    #                         'cond_hist': dp_data['cond_hist'],
    #                         'conditions': dp_data['conditions'],
    #                     })
    #
    #             file_path = os.path.join(self.cache_dir, f'{view_name}_chunk{chunk_id}.pkl')
    #             with open(file_path, 'wb') as f:
    #                 pickle.dump(graph_list, f)
    #             print(f'Saved {view_name} chunk {chunk_id} to {file_path}')
    #
    #             if view_name == 'patient_disease':
    #                 seq_path = os.path.join(self.cache_dir, f'seqdata_chunk{chunk_id}.pkl')
    #                 with open(seq_path, 'wb') as f:
    #                     pickle.dump(seqdata_chunk, f)
    #                 print(f'Saved seqdata chunk {chunk_id} to {seq_path}')


    def build_and_save_all(self):
        for view_name in self.view_config:
            print(f'Building graphs for view: {view_name}')
            graph_list = []

            for dp_data in tqdm(self.dataset, desc=f'Building {view_name}'):
                graph = self.build_single_graph(dp_data, view_name)
                graph_list.append(graph)

                # Save sequence data only once
                if view_name == 'patient_disease':
                    self.seqdata.append({
                        'adm_time': dp_data['adm_time'],
                        'procedures': dp_data['procedures'],
                        'drugs': dp_data['drugs'],
                        'cond_hist': dp_data['cond_hist'],
                        'conditions': dp_data['conditions'],
                    })

            # Save this view's graphs to a file
            file_path = os.path.join(self.cache_dir, f'{view_name}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(graph_list, f)
            print(f'Saved {view_name} graphs to {file_path}')

        # Save seqdata
        seqdata_path = os.path.join(self.cache_dir, 'seqdata.pkl')
        with open(seqdata_path, 'wb') as f:
            pickle.dump(self.seqdata, f)
        print(f'Saved sequence data to {seqdata_path}')

        vocab_path = os.path.join(self.cache_dir, 'tokenizer_vocab.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump({
                'cond_hist': self.c_tokenzier.vocabulary,
                'drugs': self.d_tokenzier.vocabulary,
                'procedures': self.p_tokenzier.vocabulary,

            }, f)
        print(f"[GraphBuilder] tokenizer vocab saved to {vocab_path}")





