"""
https://www.guyuehome.com/23747
GraphSAGE encompasses two aspects: one is the sampling of neighbors, and the other is the aggregation operation on the neighbors.
To achieve more efficient sampling, nodes and their neighboring nodes can be stored together, that is, maintaining a table that corresponds to the relationship between nodes and their neighbors.
Two functions are implemented to perform the specific operations of sampling: `sampling` is for first-order sampling, and `multihop_sampling` uses `sampling` to achieve multi-order sampling capabilities.
The result of the sampling is the ID of the nodes, and the features of each node need to be queried based on the node's ID.
"""
import numpy as np


def sampling(src_nodes, sample_num, neighbor_table):
    """Sample a specified number of neighboring nodes from the source node, noting that the sampling is with replacement;
    When the number of neighboring nodes of a certain node is less than the sampling quantity, the sampling result includes duplicate nodes.

    Arguments:
        src_nodes {list, ndarray} -- Source nodes list.
        sample_num {int} -- Number of nodes to be sampled.
        neighbor_table {dict} -- Mapping table from a node to its neighboring nodes.

    Returns:
        np.ndarray -- The list composed of the sampling results.
    """
    results = []
    for sid in src_nodes:
        # Sample with replacement from the neighbors of a node.
        res = np.random.choice(neighbor_table[sid], size=(sample_num,))
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """Perform multi-order sampling based on the source node.

    Arguments:
        src_nodes {list, np.ndarray} -- Source node id list.
        sample_nums {list of int} -- The number of samples required for each order.
        neighbor_table {dict} -- Mapping table from a node to its neighboring nodes.

    Returns:
        [list of ndarray] -- The results of sampling for each order.
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result
