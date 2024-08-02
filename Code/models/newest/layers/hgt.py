# https://github.com/pyg-team/pytorch_geometric/blob/2.1.0/examples/hetero/hgt_dblp.py
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class HGT(torch.nn.Module):
    def __init__(self, meta, n_feat_out, n_head, n_layer):
        super(HGT, self).__init__()
        n_hidden = n_feat_out
        node_types = meta[0]
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, n_hidden)

        self.convs = torch.nn.ModuleList()
        for _ in range(n_layer):
            try: # torch 1.12.1 cu102 torch-geometric==2.3.1
                conv = HGTConv(n_hidden, n_hidden, meta, n_head, group="sum")
            except:  # torch 1.12.1 cu113 torch-geometric==2.5.3
                conv = HGTConv(n_hidden, n_hidden, meta, n_head)
            self.convs.append(conv)

    def forward(self, X_dict, E_dict):
        X_dict = {
            node_type: self.lin_dict[node_type](X).relu_()
            for node_type, X in X_dict.items()
        }

        for conv in self.convs:
            X_dict = conv(X_dict, E_dict)

        return X_dict
