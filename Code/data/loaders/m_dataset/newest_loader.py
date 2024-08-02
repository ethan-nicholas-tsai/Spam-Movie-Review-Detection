import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from data.datasets.m_dataset import MaoYanDataset, DouBanLongComments
from data.transforms.m_dataset.newest.data_featurizer import NewestFeaturizer
from data.transforms.m_dataset.newest.fgraph_featurizer import FactGraphFeaturizer
from data.transforms.m_dataset.newest.rgraph_featurizer import ReviewGraphFeaturizer
from utils.data_util import DataSampler

import pickle
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


# 训练时加载数据
class NewestDataset(Dataset):
    def __init__(
        self,
        comment_path=None,
        movie_path=None,
        long_comment_path=None,
        data_load_path: dict = None,
        text_feature_path=None,
        fgraph_feature_path=None,
        rgraph_feature_path=None,
        feature_load_path: dict = None,
        feature_flag=2**2 + 2**1 + 2**1,
    ):
        """
        :param feature_flag: Use all features by default (7).
        """
        super(NewestDataset, self).__init__()
        self.comment_path = comment_path
        self.movie_path = movie_path
        self.long_comment_path = long_comment_path
        self.text_feature_path = text_feature_path
        self.fgraph_feature_path = fgraph_feature_path
        self.rgraph_feature_path = rgraph_feature_path
        self.feature_flag = feature_flag
        self.parse_module_flag()

    def parse_module_flag(self):
        """Text feature in position 0, review graph feature in position 1, factual knowledge graph feature in position 2.
        Factual knowledge graph feature: 2^2
        Review graph feature: 2^1
        Text feature：2^0
        """
        flag = self.feature_flag
        self.text_flag = flag % 2
        flag = flag // 2
        self.rgraph_flag = flag % 2
        flag = flag // 2
        self.fgraph_flag = flag
        print(
            "Factual knowledge graph feature: {}, Review graph feature: {}, Text feature: {}".format(
                self.fgraph_flag, self.rgraph_flag, self.text_flag
            )
        )
        return self.text_flag, self.rgraph_flag, self.fgraph_flag

    def load_data(self, comment_path=None, movie_path=None):
        if not comment_path:
            comment_path = self.comment_path
        if not movie_path:
            movie_path = self.movie_path
        self.maoyan_dataset = MaoYanDataset(
            comment_path=comment_path, movie_path=movie_path
        )
        commment_movie_ids_in_graph = NewestFeaturizer.get_comment_movie_graph_index(
            maoyan_dataset=self.maoyan_dataset
        )
        self.movie_ids = commment_movie_ids_in_graph
        maoyan_comments = self.maoyan_dataset.get_maoyan_comments()
        self.labels = maoyan_comments.get_comment_label()
        return self.maoyan_dataset

    def load_feature(
        self, text_feature_path=None, fgraph_feature_path=None, rgraph_feature_path=None
    ):
        if not text_feature_path:
            text_feature_path = self.text_feature_path if self.text_flag else None
        if not fgraph_feature_path:
            fgraph_feature_path = self.fgraph_feature_path if self.fgraph_flag else None
        if not rgraph_feature_path:
            rgraph_feature_path = self.rgraph_feature_path if self.rgraph_flag else None
        (
            text_feature,
            rgraph_feature,
            fgraph_feature,
        ) = NewestFeaturizer.load_feature(
            text_feature_path=text_feature_path,
            fgraph_feature_path=fgraph_feature_path,
            rgraph_feature_path=rgraph_feature_path,
        )
        self.text_feature = text_feature
        self.rgraph_feature = rgraph_feature
        self.fgraph_feature = fgraph_feature

        return text_feature, rgraph_feature, fgraph_feature

    def build_no_graph_dataset(self, pos_ratio, num_base, seed, dataset_scale=None):
        """Create a dataset without graph features.
        :pos_ratio, 0.5
        :num_base, 1000
        :seed, 2022
        :dataset_scale, 0 for the whole dataset
        :return dataset
        """
        self.review_ids = list(range(len(self.labels)))
        self.text = self.text_feature[0] if self.text_feature else self.labels
        self.mask = self.text_feature[1] if self.text_feature else self.labels
        self.review_dataset = [
            self.labels,
            self.movie_ids,
            self.text,
            self.mask,
            # self.text_feature,
            self.review_ids,
        ]
        dataset = DataSampler.bi_cls_data_pipe(
            self.review_dataset, pos_ratio, num_base, seed, dataset_scale=dataset_scale
        )
        # self.labels, self.movie_ids, self.embeddings, self.review_ids = dataset
        self.labels, self.movie_ids, self.text, self.mask, self.review_ids = dataset
        print(pos_ratio, sum(self.labels), len(self.labels))
        return dataset

    def build_graph_dataset(self, batch_size, num_neighbors=[10, 10], num_workers=0):
        """
        :batch_size
        :num_neighbors The default is [10, 10], which means that 10 neighbors are sampled for each node and the process is iterated 2 times.
        :num_workers The default is 0.
        :return dataset
        """
        data = HeteroData()
        data["review"].x = torch.tensor(self.text_feature)
        data["review"].y = torch.tensor(self.labels)
        data["review"].train_mask = torch.tensor([True] * len(self.labels))
        for k, v in self.rgraph_feature[1].items():
            data["review", k, "review"].edge_index = v

        input_nodes = ("review", data["review"].train_mask)
        dataset = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            shuffle=True,
            input_nodes=input_nodes,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return dataset

    def __getitem__(self, item):
        """
        Use `torch.tensor` on-the-fly to save memory; if you want to save time, convert to tensor in the `build_no_graph_dataset` function.
        """
        # text_features = torch.tensor(self.text_feature[item])
        text_features = [torch.tensor(self.text[item]), torch.tensor(self.mask[item])]
        review_ids = torch.tensor(self.review_ids[item])
        movie_ids = torch.tensor(self.movie_ids[item])
        labels = torch.tensor(self.labels[item])
        return labels, movie_ids, text_features, review_ids

    def __len__(self):
        return len(self.labels)


class NewestDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        val_rate=0.2,
        test_rate=0.2,
        random_noise=0,
        seed=2022,
    ):
        """
        :param dataset:
        :param batch_size:
        :param shuffle:
        :param train:
        :param rate:
        :return:
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.val_rate = val_rate
        self.test_rate = test_rate
        self.random_noise = random_noise
        self.seed = seed
        self.split_dataset()
        self.add_noise()

    def split_dataset(self):
        len_test = int(len(self.dataset) * self.test_rate)
        len_val = int(len(self.dataset) * self.val_rate)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(
            self.dataset, [len(self.dataset) - len_test - len_val, len_val, len_test]
        )
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_split_indices(self):
        return (
            self.train_dataset.indices,
            self.val_dataset.indices,
            self.test_dataset.indices,
        )

    def add_noise(self):
        if self.random_noise:
            random_data = [list(i) for i in self.train_dataset]
            data_num = len(random_data)
            data_index = list(range(data_num))
            noise_data_num = int(self.random_noise * data_num)
            random.seed(self.seed)
            random.shuffle(data_index)
            noise_data_index = data_index[:noise_data_num]
            for idx in noise_data_index:
                if random_data[idx][0] == 1:
                    random_data[idx][0] = torch.tensor(0)
                else:
                    random_data[idx][0] = torch.tensor(1)
            self.train_dataset = random_data

    def get_dataloader(self, batch_size=0):
        self.batch_size = batch_size if batch_size else self.batch_size

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
        )

        return train_loader, val_loader, test_loader
