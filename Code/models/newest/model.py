import torch
from torch import nn
from bases.model_base import Network
from models.newest.text_model import TextBModel
from models.newest.graph_model import FactGraphModel, ReviewGraphModel
from models.newest.classifier import Classifier
from torch_geometric.data import HeteroData


class TRFG(Network):
    """TRFG Model
    Graph plus multi-layer attention model.
    """

    def __init__(
        self,
        text_model_out_dim,
        text_bert_path,
        text_bert_finetune,
        text_bert_finetune_layers,
        rgraph_model_base,
        rgraph_meta,
        rgraph_data: HeteroData,
        rgraph_model_out_dim,
        fgraph_model_base,
        fgraph_meta,
        fgraph_E,
        fgraph_X,
        fgraph_model_out_dim,
        compare_network_out_dim,
        rgraph_n_hgt_head=2,
        rgraph_n_hgt_layer=2,
        rgraph_sample=True,
        rgraph_num_neighbors=[-1],
        fgraph_n_hgt_head=2,
        fgraph_n_hgt_layer=2,
        compare_flag=1,
        module_flag=2**2 + 2**1 + 2**0,
        device=None,
        TEST=False,
    ):
        super(TRFG, self).__init__(device=device)
        self.module_flag = module_flag
        self.compare_flag = compare_flag
        self.parse_module_flag()
        # The input dimension is related to the flag.
        classifier_in_dim = 0

        if self.text_flag:
            # Text model
            self.text_model = TextBModel(
                text_model_out_dim,
                text_bert_path,
                bert_finetune=text_bert_finetune,
                finetune_layers=text_bert_finetune_layers,
                device=device,
            )
            classifier_in_dim += text_model_out_dim
        if self.rgraph_flag:
            # Review graph model
            self.rgraph_model = ReviewGraphModel(
                graph_meta=rgraph_meta,
                graph_data=rgraph_data,
                out_dim=rgraph_model_out_dim,
                base_model=rgraph_model_base,
                n_head=rgraph_n_hgt_head,
                n_layer=rgraph_n_hgt_layer,
                device=device,
                sample=rgraph_sample,
                num_neighbors=rgraph_num_neighbors,
                TEST=TEST,
            )
            classifier_in_dim += rgraph_model_out_dim
        if self.fgraph_flag:
            # External factual knowledge graph
            self.fgraph_model = FactGraphModel(
                graph_meta=fgraph_meta,
                E_dict=fgraph_E,
                X_dict=fgraph_X,
                out_dim=fgraph_model_out_dim,
                base_model=fgraph_model_base,
                n_head=fgraph_n_hgt_head,
                n_layer=fgraph_n_hgt_layer,
                device=device,
            )
            if self.text_flag and self.compare_flag:
                self.linear_graph = nn.Linear(
                    fgraph_model_out_dim * 2, compare_network_out_dim
                )
                classifier_in_dim += compare_network_out_dim
            else:
                classifier_in_dim += fgraph_model_out_dim

        # Classifier
        self.classifier = Classifier(
            in_dim=classifier_in_dim,
            attention_flag=self.text_flag + self.rgraph_flag + self.fgraph_flag,
        )

    def parse_module_flag(self):
        """Text in position 0, review graph in position 1, factual knowledge graph in position 2.
        Factual knowledge graph: 2^2
        Review graph: 2^1
        Text: 2^0
        """
        flag = self.module_flag
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

    def forward(self, text_feature_in, review_ids, movie_ids, train: bool):
        """
        :param text_feature_in: Text feature
        :param review_ids: Review ids (graph feature)
        :param movie_ids: Movie ids (graph feature)
        :param train: If set for train or not
        :return:
        """
        all_feature = []
        if self.text_flag:
            text_feature_out = self.text_model(*text_feature_in) # TextBModel
            all_feature.append(text_feature_out)
        if self.rgraph_flag:
            rgraph_feature_out = self.rgraph_model(review_ids, train=train)
            all_feature.append(rgraph_feature_out)
        if self.fgraph_flag:
            fgraph_feature_out = self.fgraph_model(movie_ids)
            if self.text_flag and self.compare_flag:
                fgraph_feature_out = torch.cat(
                    (
                        (text_feature_out - fgraph_feature_out),
                        torch.mul(text_feature_out, fgraph_feature_out),
                    ),
                    1,
                )
                fgraph_feature_out = self.linear_graph(fgraph_feature_out)
            all_feature.append(fgraph_feature_out)
        all_feature_cat = torch.cat(tuple(all_feature), 1)
        # Classifier
        result = self.classifier(all_feature_cat)
        return result
