# https://zhuanlan.zhihu.com/p/412270208?utm_id=0
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GAT(torch.nn.Module):
    def __init__(self, n_feat_in, n_feat_out, heads=1):
        super(GAT, self).__init__()
        n_hid = n_feat_out
        self.gat1 = GATConv(n_feat_in, n_hid, heads=heads)
        self.gat2 = GATConv(n_hid * heads, n_feat_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X, E):
        X = self.gat1(X, E)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.gat2(X, E)
        return X
