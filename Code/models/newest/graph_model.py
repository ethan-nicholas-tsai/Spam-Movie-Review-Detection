import torch
import torch.nn as nn
from bases.model_base import Network
from models.newest.layers.hgt import HGT
from models.newest.layers.neighbor_sampler import CNeighborSampler
from torch_geometric.data import HeteroData


class FactGraphModel(Network):
    def __init__(
        self,
        graph_meta,
        E_dict: torch.tensor,  # Adjacent matrix for graph edge.
        X_dict: torch.tensor,  # Node feature
        out_dim,  # The size of the output after graph embedding.
        base_model="hgt",  # Base model.
        n_head=2,  # Number of attention head.
        n_layer=2,  # Number of layer.
        device=None,  # Run on which GPU.
    ):
        super(FactGraphModel, self).__init__(device=device)
        self.base_model = base_model
        print("Fact Graph Base Model: {}".format(self.base_model.upper()))
        self.X_dict = X_dict
        self.E_dict = E_dict
        if base_model == "hgt":
            self.hgt = HGT(
                meta=graph_meta, n_feat_out=out_dim, n_head=n_head, n_layer=n_layer
            )
        else:
            pass

    def forward(self, movie_ids):
        if self.base_model == "hgt":
            X_dict = self.hgt(self.X_dict, self.E_dict)
        else:
            pass

        feature_list = []
        for i in movie_ids:
            feature_list.append(X_dict["movie"][i])

        feature_list = torch.stack(feature_list, 0)

        return feature_list  # The results of the embeddings for all movies.


class ReviewGraphModel(Network):
    def __init__(
        self,
        graph_meta,
        graph_data: HeteroData,
        out_dim,  # The size of the output after graph embedding.
        base_model="hgt",  # Base model
        n_head=2,  # Number of attention head.
        n_layer=2,  # Number of layer
        device=None,  # Run on which GPU.
        sample=True,
        num_neighbors: list = [10, 10],  # Sample 10 neighbors for each node and perform 2 iterations.
        TEST=False,
    ):
        super(ReviewGraphModel, self).__init__(device=device)
        self.TEST = TEST
        self.device = device
        self.base_model = base_model
        print("Review Graph Base Model: {}".format(self.base_model.upper()))
        self.sample = sample
        if self.TEST:
            test_input_nodes = ("review", graph_data["review"].test_mask)
            self.neighbor_sampler_test = CNeighborSampler(
                graph_data,
                num_neighbors=num_neighbors,
                shuffle=True,
                input_nodes=test_input_nodes,
                filter_per_worker=True,
                num_workers=0,
            )
        else:
            train_input_nodes = ("review", graph_data["review"].train_mask)
            val_input_nodes = ("review", graph_data["review"].val_mask)
            self.neighbor_sampler_train = CNeighborSampler(
                graph_data,
                num_neighbors=num_neighbors,
                shuffle=True,
                input_nodes=train_input_nodes,
                filter_per_worker=True,
                num_workers=0,
            )
            self.neighbor_sampler_val = CNeighborSampler(
                graph_data,
                num_neighbors=num_neighbors,
                shuffle=True,
                input_nodes=val_input_nodes,
                filter_per_worker=True,
                num_workers=0,
            )
        self.X_dict = graph_data.x_dict
        self.E_dict = graph_data.edge_index_dict
        if base_model == "hgt":
            self.hgt = HGT(
                meta=graph_meta, n_feat_out=out_dim, n_head=n_head, n_layer=n_layer
            )
        else:
            pass

    def forward(self, review_ids, train: bool):
        if self.TEST:
            neighbor_sampler = self.neighbor_sampler_test
        else:
            neighbor_sampler = (
                self.neighbor_sampler_train if train else self.neighbor_sampler_val
            )
        feature_list = []
        if self.base_model == "hgt":
            if self.sample:
                subgraph = neighbor_sampler.collate_fn(review_ids)
                X_dict = self.hgt(
                    subgraph.x_dict,
                    subgraph.edge_index_dict,
                )
                batch_size = subgraph["review"].batch_size
                feature_list = X_dict["review"][:batch_size]
            else:
                X_dict = self.hgt(self.X_dict, self.E_dict)
                for i in review_ids:
                    feature_list.append(X_dict["review"][i])

                feature_list = torch.stack(feature_list, 0)
        else:
            pass

        return feature_list
