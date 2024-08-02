from typing import Callable, Dict, List, Optional, Tuple, Union

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.node_loader import NodeLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.typing import EdgeType, InputNodes, OptTensor


from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.mixin import AffinityMixin
from torch_geometric.loader.utils import (
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    get_input_nodes,
)
from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.typing import InputNodes, OptTensor


class CNeighborSampler:
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: InputNodes = None,
        input_id: OptTensor = None,
        input_time: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        is_sorted: bool = False,
        neighbor_sampler: Optional[NeighborSampler] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: bool = False,
        custom_cls: Optional[HeteroData] = None,
        **kwargs,
    ):
        # Get node type (or `None` for homogeneous graphs):
        try: # torch 1.12.1 cu102 torch-geometric==2.3.1
            input_type, input_nodes = get_input_nodes(data, input_nodes)
        except: # torch 1.12.1 cu113 torch-geometric==2.5.3
            input_type, input_nodes, _ = get_input_nodes(data, input_nodes)

        self.data = data
        if neighbor_sampler is None:
            self.node_sampler = NeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                directed=directed,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get("num_workers", 0) > 0,
            )

        self.input_data = NodeSamplerInput(
            input_id=input_id,
            node=input_nodes,
            time=input_time,
            input_type=input_type,
        )

        self.inverted_input_index_dict = {
            int(idx): i for i, idx in enumerate(self.input_data.node)
        }

        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        r"""Samples a subgraph from a batch of input nodes."""
        reindex = [self.inverted_input_index_dict[int(it)] for it in index]
        # input_data: NodeSamplerInput = self.input_data[index]
        input_data: NodeSamplerInput = self.input_data[reindex]

        out = self.node_sampler.sample_from_nodes(input_data)

        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            out = self.filter_fn(out)

        return out

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object to be used downstream.
        """
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        if isinstance(out, SamplerOutput):
            data = filter_data(
                self.data,
                out.node,
                out.row,
                out.col,
                out.edge,
                self.node_sampler.edge_permutation,
            )

            if "n_id" not in data:
                data.n_id = out.node
            if out.edge is not None and "e_id" not in data:
                data.e_id = out.edge

            data.batch = out.batch
            data.num_sampled_nodes = out.num_sampled_nodes
            data.num_sampled_edges = out.num_sampled_edges

            data.input_id = out.metadata[0]
            data.seed_time = out.metadata[1]
            data.batch_size = out.metadata[0].size(0)

        elif isinstance(out, HeteroSamplerOutput):
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(
                    self.data,
                    out.node,
                    out.row,
                    out.col,
                    out.edge,
                    self.node_sampler.edge_permutation,
                )
            else:  # Tuple[FeatureStore, GraphStore]
                data = filter_custom_store(
                    *self.data, out.node, out.row, out.col, out.edge, self.custom_cls
                )

            for key, node in out.node.items():
                if "n_id" not in data[key]:
                    data[key].n_id = node

            for key, edge in (out.edge or {}).items():
                if "e_id" not in data[key]:
                    data[key].e_id = edge

            data.set_value_dict("batch", out.batch)
            data.set_value_dict("num_sampled_nodes", out.num_sampled_nodes)
            data.set_value_dict("num_sampled_edges", out.num_sampled_edges)

            input_type = self.input_data.input_type
            data[input_type].input_id = out.metadata[0]
            data[input_type].seed_time = out.metadata[1]
            data[input_type].batch_size = out.metadata[0].size(0)

        else:
            raise TypeError(
                f"'{self.__class__.__name__}'' found invalid " f"type: '{type(out)}'"
            )

        return data if self.transform is None else self.transform(data)
