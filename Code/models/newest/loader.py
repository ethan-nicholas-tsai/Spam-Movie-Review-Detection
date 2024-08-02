from models.newest.model import TRFG
from data.loaders.m_dataset.newest_loader import NewestDataset
import torch
import numpy as np
from torch_geometric.data import HeteroData


def get_model(
    text_model_out_dim,
    text_bert_path,
    text_bert_finetune,
    text_bert_finetune_layers,
    fgraph_model_out_dim,
    fgraph_meta_path,
    fgraph_E_path,
    fgraph_X_path,
    fgraph_model_base,
    fgraph_n_hgt_head,
    fgraph_n_hgt_layer,
    rgraph_model_out_dim,
    rgraph_meta_path,
    rgraph_E_path,
    rgraph_X_path,
    rgraph_sample,
    rgraph_num_neighbors: list,
    rgraph_model_base,
    rgraph_n_hgt_head,
    rgraph_n_hgt_layer,
    compare_network_out_dim,
    module_flag,
    compare_flag,
    rgraph_train_mask: torch.tensor = None,
    rgraph_val_mask: torch.tensor = None,
    rgraph_test_mask: torch.tensor = None,
    check_point_path=None,
    device="cpu",
    TEST=False,
):
    # Run on which GPU.
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build model.
    feature_flag = module_flag - 1 if module_flag % 2 else module_flag
    dataset = NewestDataset(feature_flag=feature_flag)
    _, rgraph_feature, fgraph_feature = dataset.load_feature(
        fgraph_feature_path={
            "graph_meta": fgraph_meta_path,
            "graph_edges": fgraph_E_path,
            "node_feature": fgraph_X_path,
        },
        rgraph_feature_path={
            "graph_meta": rgraph_meta_path,
            "graph_edges": rgraph_E_path,
            "node_feature": rgraph_X_path,
        },
    )
    fgraph_meta, fgraph_E, fgraph_X = None, None, None
    rgraph_meta, rgraph_E, rgraph_X = None, None, None
    if dataset.fgraph_flag:
        fgraph_meta, fgraph_E, fgraph_X = fgraph_feature
        # Load to GPU
        fgraph_E = {k: torch.tensor(v).to(device) for k, v in fgraph_E.items()}
        fgraph_X = {k: torch.tensor(v).to(device) for k, v in fgraph_X.items()}
    if dataset.rgraph_feature:
        rgraph_meta, rgraph_E, rgraph_X = rgraph_feature
        rgraph_data = HeteroData()
        for k, v in rgraph_X.items():
            rgraph_data[k].x = torch.tensor(v).to(device)
        for k, v in rgraph_E.items():
            rgraph_data[k].edge_index = torch.tensor(v).to(device)
        if TEST:
            rgraph_data["review"].test_mask = rgraph_test_mask
        else:
            rgraph_data["review"].train_mask = rgraph_train_mask
            rgraph_data["review"].val_mask = rgraph_val_mask

    model = TRFG(
        text_model_out_dim=text_model_out_dim,
        text_bert_path=text_bert_path,
        text_bert_finetune=text_bert_finetune,
        text_bert_finetune_layers=text_bert_finetune_layers,
        rgraph_model_out_dim=rgraph_model_out_dim,
        rgraph_meta=rgraph_meta,
        rgraph_data=rgraph_data,
        rgraph_sample=rgraph_sample,
        rgraph_num_neighbors=rgraph_num_neighbors,
        rgraph_model_base=rgraph_model_base,
        rgraph_n_hgt_head=rgraph_n_hgt_head,
        rgraph_n_hgt_layer=rgraph_n_hgt_layer,
        fgraph_model_out_dim=fgraph_model_out_dim,
        fgraph_meta=fgraph_meta,
        fgraph_E=fgraph_E,
        fgraph_X=fgraph_X,
        fgraph_model_base=fgraph_model_base,
        fgraph_n_hgt_head=fgraph_n_hgt_head,
        fgraph_n_hgt_layer=fgraph_n_hgt_layer,
        compare_network_out_dim=compare_network_out_dim,
        module_flag=module_flag,
        compare_flag=compare_flag,
        device=device,
        TEST=TEST,
    )

    if TEST:
        # Load weight
        net = torch.load(check_point_path)
        # Load model
        model.load_state_dict(net)

    # Load model to GPU
    model = model.to(device)

    return model


if __name__ == "__main__":
    pass
