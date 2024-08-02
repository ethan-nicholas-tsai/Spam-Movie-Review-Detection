# https://zhuanlan.zhihu.com/p/54525205
import torch.nn as nn


class GCN(nn.Module):
    """
    Z = AXW
    """

    def __init__(self, A, dim_in, dim_out):
        super(GCN, self).__init__()
        self.A = A
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in // 2, bias=False)
        self.fc3 = nn.Linear(dim_in // 2, dim_out, bias=False)
        self.relu = nn.ReLU()
        # self.layer_nor = nn.LayerNorm(768)
        # self.layer_nor2 = nn.LayerNorm(384)
        # self.layer_nor3 = nn.LayerNorm(dim_out)
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)
        # self.tanh = nn.Tanh()

    def forward(self, X):
        """
        计算三层gcn
        :param X:
        :return:
        """
        X = self.fc1(self.A.mm(X))
        # X = self.layer_nor(X)
        X = self.relu(X)
        # X = self.tanh(X)
        # X = self.sigmoid(X)
        X = self.dropout(X)

        X = self.fc2(self.A.mm(X))
        # X = self.layer_nor2(X)
        X = self.relu(X)
        # X = self.tanh(X)
        # X = self.sigmoid(X)
        X = self.dropout(X)

        # X = self.layer_nor(X)
        X = self.fc3(self.A.mm(X))
        # X = F.sigmoid(X)
        # X = self.layer_nor3(X)
        # X = self.relu(X)
        # X = self.tanh(X)
        # X = self.sigmoid(X)
        X = self.dropout(X)

        return X
