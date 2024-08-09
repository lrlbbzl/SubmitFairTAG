import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

from .conv import GeneralConv

class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, n_heads, n_layers, dropout=0.2, conv_name = 'gcn', activation='relu'):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        if activation == 'relu':
            self.actication = nn.ReLU()
        self.dropout = dropout
        self.gcs.append(GeneralConv(conv_name, in_dim, n_hid, n_heads))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, n_heads))

    def forward(self, node_feature, edge_index):
        h = node_feature
        for i, conv in enumerate(self.gcs):
            h = conv(h, edge_index)
            if self.dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def test(self, node_feature, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (node_feature[edge_index[0]] * node_feature[edge_index[1]]).sum(dim=-1)
        return logits, edge_index
        
        