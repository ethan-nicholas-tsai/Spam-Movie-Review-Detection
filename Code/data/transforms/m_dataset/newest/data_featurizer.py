import numpy as np
from bases.data.transform_base import FeaturizerBase
from data.datasets.m_dataset import MaoYanDataset, DouBanLongComments
from data.transforms.m_dataset.newest.rgraph_featurizer import ReviewGraphFeaturizer
from data.transforms.m_dataset.newest.fgraph_featurizer import FactGraphFeaturizer
from data.transforms.m_dataset.newest.text_featurizer import TextFeaturizer


class NewestFeaturizer(FeaturizerBase):
    def __init__(
        self,
        comment_path=None,
        movie_path=None,
        long_comment_path=None,
        bert_vocab_path=None,
        jieba_vocab_path=None,
        text_max_seq_len=256,
        graph_max_seq_len=512,  # for fgraph
        graph_max_nodes_per_unit=3,
        split_comment_node=False,
        fgraph_bidirect=False,
        inconsistency_graph=False,  # for rgraph
        movie_view=True,
        user_view=False,
        rgraph_thresh: dict = None,
        rgraph_edge_type: list = None,
        rgraph_bidirect=True,
        feature_flag=2**2 + 2**1 + 2**0,
        bert_model_path=None,
        fgraph_save_dir=".",
        rgraph_save_dir=".",
        device="cpu",
    ):
        """
        text_max_seq_len: rgraph_max_seq_len
        graph_max_seq_len: fgraph_max_seq_len
        """
        self.comment_path = comment_path
        self.movie_path = movie_path
        self.long_comment_path = long_comment_path

        self.rgraph_save_dir = rgraph_save_dir
        self.fgraph_save_dir = fgraph_save_dir
        # bert
        self.bert_vocab_path = bert_vocab_path
        self.text_max_seq_len = text_max_seq_len
        # jieba
        self.jieba_vocab_path = jieba_vocab_path
        self.graph_max_seq_len = graph_max_seq_len
        self.bert_model_path = bert_model_path
        self.device = device
        if comment_path and movie_path and long_comment_path:
            self.load_data()
        if bert_vocab_path or jieba_vocab_path:
            self.init_featurizer()
        # Maximum number of Douban long reviews per movie.
        self.graph_max_nodes_per_unit = graph_max_nodes_per_unit
        self.split_comment = split_comment_node
        self.fgraph_bidirect = fgraph_bidirect
        self.inconsistency_graph = inconsistency_graph
        self.movie_view = movie_view
        self.user_view = user_view
        self.rgraph_thresh_dict = rgraph_thresh if rgraph_thresh else {}
        self.rgraph_edge_type = rgraph_edge_type
        self.rgraph_bidirect = rgraph_bidirect
        self.feature_flag = feature_flag
        self.parse_feature_flag()

    def parse_feature_flag(self):
        """Text feature in position 0, Review graph feature in position 1, Fact graph feature in position 2.
        Fact graph: 2^2
        Review graph: 2^1
        Text: 2^0
        """
        flag = self.feature_flag
        self.text_flag = flag % 2
        flag = flag // 2
        self.rgraph_flag = flag % 2
        flag = flag // 2
        self.fgraph_flag = flag
        print(
            "Fact graph feature: {}, Review graph feature: {}, Text feature: {}".format(
                self.fgraph_flag, self.rgraph_flag, self.text_flag
            )
        )
        return self.text_flag, self.rgraph_flag, self.fgraph_flag

    def load_data(self):
        self.maoyan_dataset = MaoYanDataset(
            comment_path=self.comment_path, movie_path=self.movie_path
        )
        self.douban_dataset = DouBanLongComments(
            douban_long_comment_file=self.long_comment_path
        )
        self.maoyan_comments = self.maoyan_dataset.get_maoyan_comments()

    def init_featurizer(self):
        self.text_featurizer = TextFeaturizer(
            bert_vocab_path=self.bert_vocab_path,
            jieba_vocab_path=self.jieba_vocab_path,
            bert_model_path=self.bert_model_path,
            device=self.device,
        )
        self.rgraph_featurizer = ReviewGraphFeaturizer(
            maoyan_dataset=self.maoyan_dataset, graph_save_dir=self.rgraph_save_dir
        )
        self.fgraph_featurizer = FactGraphFeaturizer(
            maoyan_dataset=self.maoyan_dataset,
            douban_dataset=self.douban_dataset,
            graph_save_dir=self.fgraph_save_dir,
        )

    @classmethod
    def get_comment_movie_graph_index(cls, maoyan_dataset: MaoYanDataset):
        """Retrieve the sequence number of the movie that has been reviewed.
        :param maoyan_dataset
        :return comment_movie_index_list
        """
        # movie names in dataset
        movie_names = maoyan_dataset.get_movie_names()
        movie_name_index_dic = FactGraphFeaturizer.build_movie_name_index(movie_names)
        # comment movie name column
        maoyan_commments = maoyan_dataset.get_maoyan_comments()
        comment_movie_names = maoyan_commments.get_movie_name()
        # 从fgraph_featurizer的get_movie_name_index来
        comment_movie_index_list = []
        for movie_name in comment_movie_names:
            index = movie_name_index_dic.get(movie_name, -1)
            if index != -1:
                comment_movie_index_list.append(index)

        return comment_movie_index_list

    def get_text_feature(self):
        print("Begin extracting text features.")
        comment_texts = self.maoyan_comments.get_comment_text()
        self.text, self.mask = self.text_featurizer.bert_tokenize(
            text_data=comment_texts, pad=True, max_len=self.text_max_seq_len
        )
        self.text_feature = [self.text, self.mask]
        return self.text_feature

    def get_fgraph_feature(self):
        print("Begin extracting fact graph features.")
        # build graph
        self.fgraph_featurizer.build_graph(
            max_comment_count_per_movie=self.graph_max_nodes_per_unit,
            split_comment=self.split_comment,
            comment_bidirect=self.fgraph_bidirect,
            text_featurizer=self.text_featurizer,
            seg_len=self.graph_max_seq_len,
        )
        self.fgraph_meta = self.fgraph_featurizer.get_graph_meta()
        # graph edges
        self.fgraph_edges = self.fgraph_featurizer.get_graph_edges()
        # node feature
        self.text_featurizer.set_max_seq_len(max_seq_len=self.graph_max_seq_len)
        self.fgraph_node_feature = self.fgraph_featurizer.get_node_feature(
            featurizer=self.text_featurizer
        )
        self.fgraph_feature = [
            self.fgraph_meta,
            self.fgraph_edges,
            self.fgraph_node_feature,
        ]
        return self.fgraph_feature

    def get_rgraph_feature(self):
        print("Begin extracting review graph features.")
        self.text_featurizer.set_max_seq_len(max_seq_len=self.text_max_seq_len)
        self.rgraph_node_feautre = self.rgraph_featurizer.get_node_feature(
            featurizer=self.text_featurizer
        )
        (
            self.rgraph_featurizer.build_graph(
                movie_view=self.movie_view,
                user_view=self.user_view,
                **self.rgraph_thresh_dict
            )
            if not self.inconsistency_graph
            else self.rgraph_featurizer.build_inconsistency_graph(
                movie_view=self.movie_view,
                user_view=self.user_view,
                **self.rgraph_thresh_dict
            )
        )
        self.rgraph_edges = self.rgraph_featurizer.load_graph(
            graph_base=self.rgraph_featurizer.save_dir,
            bidirect=self.rgraph_bidirect,
            edge_type=self.rgraph_edge_type,
        )
        self.rgraph_feature = [self.rgraph_edges, self.rgraph_node_feautre]
        return self.rgraph_feature

    def get_feature(self):
        # text
        text_feature = None
        if self.text_flag:
            text_feature = self.get_text_feature()
        # rgraph
        rgraph_feature = None
        if self.rgraph_flag:
            rgraph_feature = self.get_rgraph_feature()
        # fgraph
        fgraph_feature = None
        if self.fgraph_flag:
            fgraph_feature = self.get_fgraph_feature()
        print("All features have been extracted.")
        return text_feature, rgraph_feature, fgraph_feature

    def save_text_feature(self, path=None):
        np.save(path, np.array(self.text_feature))
        print("save to {}".format(path))

    def save_rgraph_feature(self, path=None):
        meta_save_path = self.rgraph_featurizer.save_graph_meta(
            path.get("graph_meta", None)
        )
        print("save to {}".format(meta_save_path))
        # Already saved during the `build_graph` process.
        # edge_save_path = self.rgraph_featurizer.save_graph(path("graph_edges", None))
        # print("save to {}".format(edge_save_path))
        feature_save_path = self.rgraph_featurizer.save_node_feature(
            path.get("node_feature", None)
        )
        print("save to {}".format(feature_save_path))

    def save_fgraph_node_data(self, path=None):
        self.fgraph_featurizer.save_node_content(path)
        print("save to {}".format(path))

    def save_fgraph_feature(self, path: dict = None):
        meta_save_path = self.fgraph_featurizer.save_graph_meta(
            path.get("graph_meta", None)
        )
        print("save to {}".format(meta_save_path))
        edge_save_path = self.fgraph_featurizer.save_graph(
            path.get("graph_edges", None)
        )
        print("save to {}".format(edge_save_path))
        feature_save_path = self.fgraph_featurizer.save_node_feature(
            path.get("node_feature", None)
        )
        print("save to {}".format(feature_save_path))

    def save_feature(
        self, text_feature_path=None, rgraph_feature_path=None, fgraph_feature_path=None
    ):
        # text
        if self.text_flag:
            text_feature_path = text_feature_path if text_feature_path else {}
            self.save_text_feature(path=text_feature_path)
        # rgraph
        if self.rgraph_flag:
            rgraph_feature_path = rgraph_feature_path if rgraph_feature_path else {}
            self.save_rgraph_feature(path=rgraph_feature_path)
        # fgraph
        if self.fgraph_flag:
            fgraph_feature_path = fgraph_feature_path if fgraph_feature_path else {}
            self.save_fgraph_feature(path=fgraph_feature_path)

    # @classmethod
    # def load_text_feature(cls, path=None):
    #     text_feature = np.load(path, allow_pickle=True).item()
    #     text_feature = text_feature["review"].tolist()
    #     return text_feature

    @classmethod
    def load_text_feature(cls, path=None):
        text_feature = np.load(path)
        text_feature = text_feature.tolist()
        return text_feature

    @classmethod
    def load_rgraph_feature(cls, path: list = None, bidirect=True, edge_type=None):
        graph_meta = None
        graph_edges = None
        node_feature = None
        if path.get("graph_meta"):
            graph_meta = ReviewGraphFeaturizer.load_graph_meta(path["graph_meta"])
        if path.get("graph_edges"):
            graph_edges = ReviewGraphFeaturizer.load_graph(
                graph_base=path["graph_edges"], bidirect=bidirect, edge_type=edge_type
            )
        if path.get("node_feature"):
            node_feature = ReviewGraphFeaturizer.load_node_feature(
                path=path["node_feature"]
            )
        graph_feature = [graph_meta, graph_edges, node_feature]
        return graph_feature

    @classmethod
    def load_fgraph_node_data(cls, path=None):
        graph_node_data = FactGraphFeaturizer.load_node_content(path)
        return graph_node_data

    @classmethod
    def load_fgraph_feature(cls, path: dict = None):
        graph_meta = FactGraphFeaturizer.load_graph_meta(path["graph_meta"])
        graph_edges = FactGraphFeaturizer.load_graph(path["graph_edges"])
        node_feature = FactGraphFeaturizer.load_node_feature(path["node_feature"])
        graph_feature = [graph_meta, graph_edges, node_feature]
        return graph_feature

    @classmethod
    def load_feature(
        cls,
        text_feature_path=None,
        rgraph_feature_path=None,
        fgraph_feature_path=None,
    ):
        text_feature = (
            cls.load_text_feature(text_feature_path) if text_feature_path else None
        )
        rgraph_feature = (
            cls.load_rgraph_feature(rgraph_feature_path)
            if rgraph_feature_path
            else None
        )
        fgraph_feature = (
            cls.load_fgraph_feature(fgraph_feature_path)
            if fgraph_feature_path
            else None
        )
        return text_feature, rgraph_feature, fgraph_feature
