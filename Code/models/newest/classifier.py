import torch
from torch import nn
from models.newest.layers.attention import SelfAttention


class Classifier(nn.Module):
    def __init__(self, in_dim, attention_flag):  # text, rgraph, fgraph model的out_dim之和
        super(Classifier, self).__init__()
        self.attention_flag = attention_flag
        if self.attention_flag > 1:
            print("Attention Flag: {}".format(self.attention_flag))
            if in_dim % self.attention_flag != 0:
                raise ValueError("The output features from the contrast network must be consistent across text, Review graph, and Fact graph!")
            self.in_dim = in_dim
            self.in_dim_per_feat = in_dim // self.attention_flag
            self.att = SelfAttention(
                input_size=self.in_dim_per_feat,
                hidden_size=self.in_dim_per_feat,
                num_attention_heads=2,
                hidden_dropout_prob=0.4,
            )

        hidden_dim_1 = in_dim // 2
        hidden_dim_2 = in_dim // (2**2)
        self.l1 = nn.Linear(in_dim, hidden_dim_1)
        self.batch_nor = nn.BatchNorm1d(hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.batch_nor2 = nn.BatchNorm1d(hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.l3 = nn.Linear(hidden_dim_2, 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        if self.attention_flag > 1:
            batch_size = X.shape[0]
            X = X.reshape(batch_size, -1, self.in_dim_per_feat)
            X = self.att(X)
            X = X.reshape(batch_size, self.in_dim)

        X = self.l1(X)
        X = self.batch_nor(X)
        X = self.relu1(X)
        X = self.dropout(X)

        X = self.l2(X)
        X = self.batch_nor2(X)
        X = self.relu2(X)
        X = self.dropout(X)

        X = self.l3(X)
        X = self.sigmoid(X)
        return X
