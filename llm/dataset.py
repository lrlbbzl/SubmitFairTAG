import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

NUM_NEG_PER_SAMPLE = 100

class LPDataset(Dataset):
    def __init__(self, input_ids, attention_mask, pos_edge, neg_edge, features=None, oracle_edges=None):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.pos_edge = pos_edge
        self.neg_edge = neg_edge
        self.all_edge = torch.cat([self.pos_edge, self.neg_edge], dim=-1) # (2, N)
        self.labels = torch.FloatTensor([1] * self.pos_edge.size(1) + [0] * self.neg_edge.size(1))
        if features is not None:
            src_fea, dst_fea = features[self.all_edge[0, :]], features[self.all_edge[1, :]]
            self.is_heterogeneous = torch.tensor((src_fea != dst_fea), dtype=torch.long).cpu()

        if oracle_edges is not None:
            self.oracle_edges = oracle_edges
            
    def __len__(self):
        return self.all_edge.size(1)
    
    def __getitem__(self, idx):
        src, dst, l = self.all_edge[0][idx], self.all_edge[1][idx], self.labels[idx]
        node_idx = torch.LongTensor([src, dst])
        return_map = {
            'input_ids' : self.input_ids[node_idx],
            'attention_mask' : self.attention_mask[node_idx], 
            'label' : l,
            'node_idx' : node_idx
        }
        if hasattr(self, 'is_heterogeneous'):
            return_map.update({'is_heterogeneous' : self.is_heterogeneous[idx]})
        if hasattr(self, 'oracle_edges'):
            src, dst = self.oracle_edges[0][idx], self.oracle_edges[1][idx]
            oracle_node_idx = torch.LongTensor([src, dst])
            return_map.update({'oracle_input_ids' : self.input_ids[oracle_node_idx], 'oracle_attention_mask' : self.attention_mask[oracle_node_idx]})
        return return_map


class PoDataset(Dataset):
    def __init__(self, input_ids, attention_mask, po_edges):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.po_edges = po_edges

    def __len__(self, ):
        return self.po_edges.size(1)
    
    def __getitem__(self, idx):
        src, pre, aft = self.po_edges[0][idx], self.po_edges[1][idx], self.po_edges[2][idx]
        types = self.po_edges[3][idx]
        pre_node_idx = torch.LongTensor([src, pre])
        pre_node_idx = torch.LongTensor([src, aft])
        return {
            'pre_input_ids' : self.input_ids[pre_node_idx],
            'pre_attention_mask' : self.attention_mask[pre_node_idx],
            'aft_input_ids' : self.input_ids[pre_node_idx],
            'aft_attention_mask' : self.attention_mask[pre_node_idx],
            'types' : types
        }
    

class EncodeDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self, ):
        return self.input_ids.size(0)
    
    def __getitem__(self, index):
        return {
            'input_ids' : self.input_ids[index],
            'attention_mask' : self.attention_mask[index]
        }