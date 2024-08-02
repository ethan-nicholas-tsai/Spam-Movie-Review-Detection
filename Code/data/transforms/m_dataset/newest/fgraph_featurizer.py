import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from data.datasets.m_dataset import MaoYanDataset
from data.datasets.m_dataset import DouBanLongComments
from bases.data.transform_base import (
    FeaturizerBase,
)  
from data.transforms.m_dataset.gmk.text_featurizer import TextFeaturizer
from pytorch_pretrained_bert import BertModel


class HGraph:
    """Heterogeneous graph construction
    Movies of the same genre are connected bidirectionally; long reviews of the same movie are connected bidirectionally; segments of long reviews are connected unidirectionally from front to back.
    """

    def __init__(self):
        self.node_type = ["movie", "comment"]
        self.relation_type = [
            ("movie", "to", "movie"),
            ("comment", "to", "movie"),
            ("movie", "to", "comment"),
            ("comment", "to", "comment"),
        ]
        self.G = {relation: set([]) for relation in self.relation_type}
        self.type_node_dic = {}  # This dictionary records the correspondence between movie genres and nodes.

    def add_movie_node(self, movie_name_index, movie_types):
        """Movies of the same type are bidirectionally connected to each other.
        :param movie_index: The sequence number corresponding to the movie.
        :param movie_types: The category or genre of a movie.
        :return:
        """
        node_to_add = movie_name_index  # Pending movie nodes to be added.
        zihuan_flag = False
        # Add to the bidirectional edges of the same type of movie.
        for movie_type in movie_types:  # For all the movie genres of `node_to_add`.
            # Find nodes of the same movie genre.
            same_type_node_set = self.type_node_dic.get(movie_type, set([]))
            # Regardless of whether it's found or not, create the movie genre and add the node (this step must be placed here, as it will create self-loops!!).
            same_type_node_set.add(node_to_add)
            self.type_node_dic[movie_type] = same_type_node_set
            # Add self-loops.
            if not zihuan_flag:
                self.G[("movie", "to", "movie")].add((node_to_add, node_to_add))
                zihuan_flag = True
            # Then establish bidirectional edges from `node_to_add` to all movies of the same type.
            for node_exist in list(same_type_node_set):
                if node_exist != node_to_add:
                    self.G[("movie", "to", "movie")].add((node_to_add, node_exist))
                    self.G[("movie", "to", "movie")].add((node_exist, node_to_add))

    def add_long_comment_node(
        self, comment_index: int, movie_index: int, bidirect=True
    ):
        """Add long review nodes.
        Long reviews of the movie are bidirectionally connected to the movie.
        :param comment_index: Index of review
        :param movie_index: Index of movie
        :param bidirect: Whether to add bidirectional connections.
        :return:
        """
        # Bidirectionally link between the movie and review.
        self.G[("comment", "to", "movie")].add((comment_index, movie_index))
        if bidirect:
            self.G[("movie", "to", "comment")].add((movie_index, comment_index))

    def add_long_comment_segment_nodes(
        self, segment_index_list: list, movie_index: int, bidirect=True
    ):
        """
        Add long review nodes.
        Long reviews of the movie are bidirectionally connected to the movie.
        Each segment of the review is unidirectionally connected from the previous segment to the next segment.
        :param segment_index_list: Index of review segment
        :param movie_index: Index of movie
        :param bidirect: Whether to add bidirectional connections.
        :return:
        """
        last_seg_index = -1
        for seg_index in segment_index_list:
            # Bidirectionally link between the movie and review.
            self.G[("comment", "to", "movie")].add((seg_index, movie_index))
            self.G[("movie", "to", "comment")].add((movie_index, seg_index))
            if last_seg_index != -1:
                # Unidirectional connection between adjacent segments.
                self.G[("comment", "to", "comment")].add((last_seg_index, seg_index))
                # Bidirectional connection between adjacent segments.
                if bidirect:
                    self.G[("comment", "to", "comment")].add(
                        (seg_index, last_seg_index)
                    )
            last_seg_index = seg_index

    def get_graph(self):
        """Retrieve the structure of the heterogeneous graph."""
        E = {}
        for k, v in self.G.items():
            v = list(v)
            v.sort()
            E[k] = np.array(v).T
        return E

    def get_meta(self):
        """Retrieve the node and edge type information of the heterogeneous graph."""
        meta = {"node": self.node_type, "relation": self.relation_type}
        return meta


class FactGraphFeaturizer(HGraph):
    """Maoyan Movie Background Knowledge Graph (Maoyan movie synopsis and Douban long reviews)."""

    def __init__(
        self,
        maoyan_dataset=None,
        douban_dataset=None,
        comment_path=None,
        movie_path=None,
        long_comment_path=None,
        graph_save_dir=".",
    ):
        """
        :param comment_path
        :param movie_path
        :param long_comment_path
        """
        super(FactGraphFeaturizer, self).__init__()
        if not maoyan_dataset:
            # TODO: raise error if comment_path or movie_path is None
            maoyan_dataset = MaoYanDataset(
                comment_path=comment_path, movie_path=movie_path
            )
        if not douban_dataset:
            douban_dataset = DouBanLongComments(
                douban_long_comment_file=long_comment_path
            )
        self.maoyan_dataset = maoyan_dataset
        self.douban_dataset = douban_dataset
        self.movie_names = self.maoyan_dataset.get_movie_names()
        self.movie_name_index_dic = FactGraphFeaturizer.build_movie_name_index(
            self.movie_names
        )
        self.movie_types = self.maoyan_dataset.get_movie_types()
        self.movie_type_index_dic = FactGraphFeaturizer.build_movie_type_index(
            self.movie_types
        )
        self.set_save_dir(graph_save_dir=graph_save_dir)

    def set_save_dir(self, graph_save_dir):
        self.save_dir = graph_save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @classmethod
    def build_movie_name_index(cls, movie_names):
        """Establish an index for movie titles."""
        movie_name_index_dic = {}
        for idx in range(len(movie_names)):
            movie_name_index_dic[movie_names[idx]] = idx  # {电影：序号}
        # for idx in range(1, 1 + len(movie_names)):
        #     movie_name_index_dic[movie_names[idx - 1]] = idx  # {电影：序号}
        return movie_name_index_dic

    @classmethod
    def build_movie_type_index(cls, movie_types):
        """Establish an index for movie types."""
        movie_type_index_dic = {}
        for idx in range(len(movie_types)):
            movie_type_index_dic[movie_types[idx]] = idx  # {电影类型：序号}
        return movie_type_index_dic

    def get_movie_type_index(self, movie_type_list, movie_type_index_dic=None):
        """ """
        if not movie_type_index_dic:
            movie_type_index_dic = self.movie_type_index_dic
        movie_type_index_list = []
        for movie_type in movie_type_list:
            index = movie_type_index_dic.get(movie_type, -1)
            if index != -1:
                movie_type_index_list.append(index)

        return movie_type_index_list

    def get_movie_name_index(self, movie_name_list, movie_name_index_dic=None):
        """Replace the movie names with sequence numbers."""
        if not movie_name_index_dic:
            movie_name_index_dic = self.movie_name_index_dic
        movie_index_list = []
        for movie_name in movie_name_list:
            index = movie_name_index_dic.get(movie_name, -1)
            if index != -1:
                movie_index_list.append(index)

        return movie_index_list

    def build_graph(
        self,
        max_comment_count_per_movie=20,
        split_comment=False,
        comment_bidirect=False,
        text_featurizer: TextFeaturizer = None,
        seg_len=512,
    ):
        """Construct a heterogeneous knowledge graph for movie backgrounds.
        :param max_comment_count_per_movie
            default 20,
            if 0, no limits of comment counts then
        :return None
        """
        if not split_comment:
            self.relation_type = self.relation_type[:-1]
            self.G.pop(("comment", "to", "comment"))
        movie_count = len(self.movie_names)  # Number of movies
        long_comment_count = 0  # Number of long reviews
        all_movie = []  # Store movie synopsis
        all_comment = []  # Store movie reviews
        all_node_info = []

        # Connect nodes of the same movie genre.
        for movie_name, idx in self.movie_name_index_dic.items():
            # print(movie_name)

            movie_type_list = self.maoyan_dataset.get_movie_type_by_name(
                movie_name=movie_name
            )  # The genre of the movie being reviewed.

            self.add_movie_node(idx, movie_type_list)  # Connect movies of the same type.

            # Add Maoyan movie synopsis.
            movie_synopsis = self.maoyan_dataset.get_movie_synopsis_by_name(movie_name)
            movie_publish_time = self.maoyan_dataset.get_movie_publish_time_by_name(
                movie_name=movie_name
            )
            all_movie.append(movie_synopsis)
            all_node_info.append(
                ["movie", movie_name, movie_synopsis, movie_publish_time]
            )

        # Comments are connected to the movie.
        for movie_name, movie_index in self.movie_name_index_dic.items():
            # Retrieve the Douban long reviews of the movie.
            comment_text_list = self.douban_dataset.get_comment_text_by_name(
                movie_name, comment_count=max_comment_count_per_movie
            )

            for comment_text in comment_text_list:
                if split_comment:
                    comment_segs = text_featurizer.bert_tokenizer.segment_text(
                        text=comment_text, seg_len=seg_len
                    )
                    seg_index_list = []
                    for i in range(len(comment_segs)):
                        seg_text = comment_segs[i]
                        seg_index = long_comment_count  # Long review segment numbering.
                        seg_index_list.append(seg_index)
                        all_comment.append(seg_text)
                        long_comment_count += 1
                    all_node_info.append(["comment", movie_name, comment_text, 0])
                    self.add_long_comment_segment_nodes(
                        segment_index_list=seg_index_list, movie_index=movie_index
                    )
                else:
                    comment_index = long_comment_count  # Long review numbering
                    self.add_long_comment_node(
                        comment_index=comment_index,
                        movie_index=movie_index,
                        bidirect=comment_bidirect,
                    )
                    all_comment.append(comment_text)
                    # Add the movie Douban reviews into the comprehensive table.
                    all_node_info.append(["comment", movie_name, comment_text, 0])
                    long_comment_count += 1
        # Convert the `node_content` into a dataframe format.
        df = pd.DataFrame(all_node_info)
        df.columns = ["节点类型", "名字", "内容", "上映时间"]
        self.graph_node_content = df
        self.movie_node_data = all_movie
        self.comment_node_data = all_comment

    def get_graph_edges(self):
        """Obtain the edges of the heterogeneous graph."""
        E = super().get_graph()
        return E

    def save_graph(self, save_path=None):
        if not save_path:
            save_path = os.path.join(self.save_dir, "fgraph_edges")
        E = self.get_graph_edges()
        np.save(save_path, E)
        return save_path

    @classmethod
    def load_graph(cls, path=None):
        if not path:
            path = os.path.join(cls.save_dir, "fgraph_edges.npy")
        E = np.load(path, allow_pickle=True).item()
        return E

    def save_node_content(self, save_path=None):
        """Save the content of the nodes in the graph (Maoyan movie synopsis, Douban long review texts, and some metadata)."""
        if not save_path:
            save_path = os.path.join(self.save_dir, "fgraph_node_content.csv")
        self.graph_node_content.to_csv(save_path, encoding="utf_8_sig")

    @classmethod
    def load_node_content(cls, path=None):
        if not path:
            path = os.path.join(cls.save_dir, "fgraph_node_content.csv")
        df = pd.read_csv(path)
        return df

    def get_node_text_feature(self, featurizer: TextFeaturizer, node_text):
        # The BERT semantic space should use the same tokenizer and vocabulary numbering; it should not use the segmentation results from Jieba.
        # node_text_feature = featurizer.jieba_tokenize(text_data=node_text, pad=True)
        batch = 8  # TODO: Adapt the batch size automatically based on the GPU size. Or use a try-except mechanism to continuously try and adjust.
        all_num = len(node_text)
        rnd = all_num // batch
        res = all_num / batch
        if res > rnd:
            rnd += 1
        batch_text_feat_list = []
        for i in range(rnd):
            # print(i, rnd)
            # print(batch * i, batch * i + 4)
            batch_text = node_text[batch * i : batch * (i + 1)]
            batch_text_tokens, batch_text_masks = featurizer.bert_tokenize(
                text_data=batch_text, pad=True
            )
            batch_text_feature = featurizer.bert_embedding(
                bert_token=batch_text_tokens, bert_mask=batch_text_masks
            )
            batch_text_feat_list.append(batch_text_feature)
        node_text_feature = np.concatenate(batch_text_feat_list, axis=0)
        # node_text_tokens, node_text_masks = featurizer.bert_tokenize(
        #     text_data=node_text, pad=True
        # )
        # node_text_feature = featurizer.bert_embedding(
        #     bert_token=node_text_tokens, bert_mask=node_text_masks
        # )
        return node_text_feature

    def get_movie_node_feature(self, featurizer: TextFeaturizer, movie_node_data):
        """Extract features of the movie nodes.
        :param movie_node_data: df
        :return movie_node_feature
        """
        movie_node_feature = self.get_node_text_feature(
            featurizer=featurizer, node_text=movie_node_data
        )
        return movie_node_feature

    def get_comment_node_feature(self, featurizer: TextFeaturizer, comment_node_data):
        """Extract features of the movie long reviews.
        :param comment_node_data: df
        :return comment_node_feature
        """
        comment_node_feature = self.get_node_text_feature(
            featurizer=featurizer, node_text=comment_node_data
        )
        return comment_node_feature

    def get_node_feature(self, featurizer: TextFeaturizer, node_data_dict=None):
        if not node_data_dict:
            node_data_dict = {
                "movie": self.movie_node_data,
                "comment": self.comment_node_data,
            }
        self.node_feature = {}
        self.node_feature["movie"] = self.get_movie_node_feature(
            featurizer, node_data_dict["movie"]
        )
        self.node_feature["comment"] = self.get_comment_node_feature(
            featurizer, node_data_dict["comment"]
        )
        return self.node_feature

    def save_node_feature(self, path=None):
        """Save the features of the graph nodes."""
        if not path:
            path = os.path.join(self.save_dir, "fgraph_node_feature")
        np.save(path, self.node_feature)
        return path

    @classmethod
    def load_node_feature(cls, path=None):
        """Load the features of the graph nodes."""
        if not path:
            path = os.path.join(cls.save_dir, "fgraph_node_feature.npy")
        node_feature = np.load(path, allow_pickle=True).item()
        return node_feature

    def get_graph_meta(self):
        """Retrieve the meta information of the heterogeneous graph."""
        meta = super().get_meta()
        return meta

    def save_graph_meta(self, save_path=None):
        if not save_path:
            save_path = os.path.join(self.save_dir, "fgraph_meta")
        meta = self.get_graph_meta()
        np.save(save_path, meta)
        return save_path

    @classmethod
    def load_graph_meta(cls, path=None):
        if not path:
            path = os.path.join(cls.save_dir, "fgraph_meta.npy")
        meta = np.load(path, allow_pickle=True).item()
        meta = (meta["node"], meta["relation"])
        return meta
