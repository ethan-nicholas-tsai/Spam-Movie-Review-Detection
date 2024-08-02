import pandas as pd
import os
from copy import deepcopy

# from tests import ti
import torch
import time
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from data.datasets.m_dataset import MaoYanDataset
from data.datasets.m_dataset import DouBanLongComments
from bases.data.transform_base import FeaturizerBase  
from data.transforms.m_dataset.gmk.text_featurizer import TextFeaturizer
from pytorch_pretrained_bert import BertModel


class ReviewGraphFeaturizer:
    """Maoyan Review Graph"""

    relation_types = []

    def __init__(
        self,
        maoyan_dataset=None,
        comment_path=None,
        movie_path=None,
        graph_save_dir=".",
    ):
        """
        :param comment_path
        :param movie_path
        """
        super(ReviewGraphFeaturizer, self).__init__()
        if not maoyan_dataset:
            # TODO: raise error if comment_path or movie_path is None
            maoyan_dataset = MaoYanDataset(
                comment_path=comment_path, movie_path=movie_path
            )

        self.maoyan_dataset = maoyan_dataset
        self.set_save_dir(graph_save_dir=graph_save_dir)

    def set_save_dir(self, graph_save_dir):
        self.save_dir = graph_save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def build_graph(
        self,
        movie_view=True,
        user_view=True,
        text_t=0.3,
        score_t=0.2,
        day_t=7,
        thumb_t=0.2,
        folcmt_t=0.2,
        urank_t=0.2,
        utowatch_t=0.2,
        uwatched_t=0.2,
        ucmt_t=0.2,
        utopic_t=0.2,
    ):
        """Connect edges"""

        with open(os.path.join(self.save_dir, "info.txt"), "w") as fin:
            if movie_view:
                fin.write("movie\n")
            if user_view:
                fin.write("user\n")
            fin.write("text_t: {}\n".format(text_t))
            fin.write("score_t: {}\n".format(score_t))
            fin.write("day_t: {}\n".format(day_t))

        node_text_feature = torch.tensor(self.node_feature["review"])

        maoyan_comments = self.maoyan_dataset.get_maoyan_comments()
        comment_df = maoyan_comments.get_data()
        maoyan_movies = self.maoyan_dataset.get_maoyan_movies()
        movie_names = maoyan_movies.get_movie_name()
        comment_movie_name_list = np.array(maoyan_comments.get_movie_name())
        # comment meta
        comment_score_list = np.array(maoyan_comments.get_comment_score())
        review_time_list = np.array(maoyan_comments.get_comment_time())
        # movie meta
        movie_scores = np.array(maoyan_movies.get_movie_score())
        movie_publish_times = np.array(maoyan_movies.get_movie_publish_time())

        def dev_set_map(r_value: float, avg_value: float, threshold: float):
            dev_value = r_value / avg_value - 1
            if dev_value < -threshold:
                return -1
            elif dev_value > threshold:
                return 1
            else:
                return 0

        def time_set_map(r_time: str, m_time: int, threshold=7):
            movie_timeStamp = int(m_time / 1000)
            # Convert to a time array.
            user_timeArray = time.strptime(r_time, "%Y-%m-%d %H:%M:%S")
            # movie_timeArray = time.strptime(movie_time, "%Y/%m/%d %H:%M")
            # Convert to a timestamp.
            user_timeStamp = int(time.mktime(user_timeArray))
            # movie_timeStamp = int(time.mktime(movie_timeArray))
            sub_timeStamp = user_timeStamp - movie_timeStamp
            # Before release.
            if sub_timeStamp < 0:
                return -1
            # Release week.
            elif sub_timeStamp / 86400 <= threshold:
                return 1
            # Other
            else:
                return 0

        movie_idx_list = []
        for mv in comment_movie_name_list:
            movie_idx_list.append(movie_names.index(mv))
        movie_idx_list = np.array(movie_idx_list)

        graph_edges = {}

        all_cmt = len(comment_score_list)

        for i in range(len(comment_movie_name_list)):
            # if i and not i % 10:
            #     print(i, all_cmt, flush=True)
            idx_i = movie_idx_list[i]
            # text_t
            emb_i = node_text_feature[i]
            # score_t
            score_set_i = dev_set_map(
                comment_score_list[i], movie_scores[idx_i], score_t
            )
            # day_t
            time_set_i = time_set_map(
                review_time_list[i], movie_publish_times[idx_i], threshold=day_t
            )
            for j in range(len(comment_movie_name_list)):
                if j <= i:
                    continue
                set_map = {}
                idx_j = movie_idx_list[j]
                # movie view
                if movie_view and idx_i == idx_j:
                    # text_t
                    emb_j = node_text_feature[j]
                    emb_sim = torch.cosine_similarity(emb_i, emb_j, dim=0)
                    # score_t
                    score_set_j = dev_set_map(
                        comment_score_list[j], movie_scores[idx_j], score_t
                    )
                    # set_map["score"] = 1 if score_set_i == score_set_j else 0
                    if score_set_i == score_set_j:
                        set_map["score"] = 1
                    elif score_set_i * score_set_j == -1:
                        set_map["score"] = 0
                    # day_t
                    time_set_j = time_set_map(
                        review_time_list[j], movie_publish_times[idx_j], threshold=day_t
                    )
                    if time_set_i * time_set_j:
                        set_map["time"] = 1
                    else:
                        set_map["time"] = 0
                else:
                    continue

                # add edge
                if emb_sim > text_t:
                    for k_a, v_a in set_map.items():
                        if k_a in ["score", "time"]:
                            edge_type = "text" + "-" + k_a
                            edge_type = (
                                edge_type + "@sim" if v_a else edge_type + "@dif"
                            )
                        else:
                            edge_type = k_a
                        fhandler = graph_edges.get(edge_type, None)
                        if not fhandler:
                            f = open(
                                os.path.join(self.save_dir, edge_type + ".txt"),
                                "w+",
                            )
                            graph_edges[edge_type] = f
                            fhandler = f
                        fhandler.write("{},{}\n".format(i, j))

    def build_inconsistency_graph(
        self,
        movie_view=True,
        user_view=True,
        text_t=0.3,
        score_t=0.2,
        day_t=7,
        thumb_t=0.2,
        folcmt_t=0.2,
        urank_t=0.2,
        utowatch_t=0.2,
        uwatched_t=0.2,
        ucmt_t=0.2,
        utopic_t=0.2,
    ):
        """Connect edges"""
        info = ""
        if movie_view:
            info += "-movie"
        if user_view:
            info += "-user"

        with open(os.path.join(self.save_dir, "info.txt"), "w") as fin:
            fin.write(info)

        node_text_feature = torch.tensor(self.node_feature["review"])

        maoyan_comments = self.maoyan_dataset.get_maoyan_comments()
        comment_df = maoyan_comments.get_data()
        maoyan_movies = self.maoyan_dataset.get_maoyan_movies()
        movie_names = maoyan_movies.get_movie_name()
        comment_movie_name_list = np.array(maoyan_comments.get_movie_name())
        # comment meta
        comment_score_list = np.array(maoyan_comments.get_comment_score())
        review_time_list = np.array(maoyan_comments.get_comment_time())
        thumbs_up_list = np.array(maoyan_comments.get_thumbs_up())
        follow_comment_number_list = np.array(
            maoyan_comments.get_follow_comment_number()
        )
        # movie meta
        movie_scores = np.array(maoyan_movies.get_movie_score())
        movie_publish_times = np.array(maoyan_movies.get_movie_publish_time())
        # user meta
        user_rank_list = np.array(maoyan_comments.get_user_rank())
        user_to_watch_number_list = np.array(maoyan_comments.get_user_to_watch_number())
        user_watched_number_list = np.array(maoyan_comments.get_user_watched_number())
        user_comment_number_list = np.array(maoyan_comments.get_user_comment_number())
        user_topic_number_list = np.array(maoyan_comments.get_user_topic_number())
        # node set
        comment_df["thumb_avg"] = comment_df.groupby("名称")["点赞"].transform("mean")
        thumb_avg_list = np.array(comment_df["thumb_avg"].to_list())
        comment_df["folcmt_avg"] = comment_df.groupby("名称")["评论的评论"].transform("mean")
        folcmt_avg_list = np.array(comment_df["folcmt_avg"].to_list())
        urank_avg = mean(user_rank_list)
        utowatch_avg = mean(user_to_watch_number_list)
        uwatched_avg = mean(user_watched_number_list)
        ucmt_avg = mean(user_comment_number_list)
        utopic_avg = mean(user_topic_number_list)

        def dev_set_map(r_value: float, avg_value: float, threshold: float):
            dev_value = r_value / avg_value - 1
            if dev_value < -threshold:
                return -1
            elif dev_value > threshold:
                return 1
            else:
                return 0

        def time_set_map(r_time: str, m_time: int, threshold=7):
            movie_timeStamp = int(m_time / 1000)
            # Convert to a time array.
            user_timeArray = time.strptime(r_time, "%Y-%m-%d %H:%M:%S")
            # movie_timeArray = time.strptime(movie_time, "%Y/%m/%d %H:%M")
            # Convert to a timestamp.
            user_timeStamp = int(time.mktime(user_timeArray))
            # movie_timeStamp = int(time.mktime(movie_timeArray))
            sub_timeStamp = user_timeStamp - movie_timeStamp
            # Before release.
            if sub_timeStamp < 0:
                return -1
            # Release week.
            elif sub_timeStamp / 86400 <= threshold:
                return 1
            # Other.
            else:
                return 0

        movie_idx_list = []
        for mv in comment_movie_name_list:
            movie_idx_list.append(movie_names.index(mv))
        movie_idx_list = np.array(movie_idx_list)

        graph_edges = {}

        all_cmt = len(comment_score_list)

        for i in range(len(comment_movie_name_list)):
            # if i and not i % 10:
            #     print(i, all_cmt, flush=True)
            idx_i = movie_idx_list[i]
            # text_t
            emb_i = node_text_feature[i]
            # score_t
            score_set_i = dev_set_map(
                comment_score_list[i], movie_scores[idx_i], score_t
            )
            # day_t
            time_set_i = time_set_map(
                review_time_list[i], movie_publish_times[idx_i], threshold=day_t
            )
            # thumb_t
            thumb_set_i = dev_set_map(thumbs_up_list[i], thumb_avg_list[i], thumb_t)
            # folcmt_t
            folcmt_set_i = dev_set_map(
                follow_comment_number_list[i], folcmt_avg_list[i], folcmt_t
            )
            # urank_t
            urank_set_i = dev_set_map(user_rank_list[i], urank_avg, urank_t)
            # utowatch_t
            utowatch_set_i = dev_set_map(
                user_to_watch_number_list[i], utowatch_avg, utowatch_t
            )
            # uwatched_t
            uwatched_set_i = dev_set_map(
                user_watched_number_list[i], uwatched_avg, uwatched_t
            )
            # ucmt_t
            ucmt_set_i = dev_set_map(user_comment_number_list[i], ucmt_avg, ucmt_t)
            # utopic_t
            utopic_set_i = dev_set_map(user_topic_number_list[i], utopic_avg, utopic_t)
            for j in range(len(comment_movie_name_list)):
                if j <= i:
                    continue
                set_map = {}
                idx_j = movie_idx_list[j]
                # movie view
                if movie_view and idx_i == idx_j:
                    # text_t
                    emb_j = node_text_feature[j]
                    emb_sim = torch.cosine_similarity(emb_i, emb_j, dim=0)
                    set_map["text"] = 1 if emb_sim > text_t else 0
                    # score_t
                    score_set_j = dev_set_map(
                        comment_score_list[j], movie_scores[idx_j], score_t
                    )
                    if score_set_i == score_set_j:
                        set_map["score"] = 1
                    elif score_set_i * score_set_j == -1:
                        set_map["score"] = 0
                    # day_t
                    time_set_j = time_set_map(
                        review_time_list[j], movie_publish_times[idx_j], threshold=day_t
                    )
                    if time_set_i == time_set_j:
                        set_map["time"] = 1
                    elif time_set_i * time_set_j == -1:
                        set_map["time"] = 0
                    # thumb_t
                    thumb_set_j = dev_set_map(
                        thumbs_up_list[j], thumb_avg_list[j], thumb_t
                    )
                    if thumb_set_i == thumb_set_j:
                        set_map["thumb"] = 1
                    elif thumb_set_i * thumb_set_j == -1:
                        set_map["thumb"] = 0
                    # folcmt_t
                    folcmt_set_j = dev_set_map(
                        follow_comment_number_list[j], folcmt_avg_list[j], folcmt_t
                    )
                    if folcmt_set_i == folcmt_set_j:
                        set_map["folcmt"] = 1
                    elif folcmt_set_i * folcmt_set_j == -1:
                        set_map["folcmt"] = 0
                # user view
                if user_view:
                    # urank_t
                    urank_set_j = dev_set_map(user_rank_list[j], urank_avg, urank_t)
                    if urank_set_i == urank_set_j:
                        set_map["urank"] = 1
                    elif urank_set_i * urank_set_j == -1:
                        set_map["urank"] = 0
                    # utowatch_t
                    utowatch_set_j = dev_set_map(
                        user_to_watch_number_list[j], utowatch_avg, utowatch_t
                    )
                    if utowatch_set_i == utowatch_set_j:
                        set_map["utowatch"] = 1
                    elif utowatch_set_i * utowatch_set_j == -1:
                        set_map["utowatch"] = 0
                    # uwatched_t
                    uwatched_set_j = dev_set_map(
                        user_watched_number_list[j], uwatched_avg, uwatched_t
                    )
                    if uwatched_set_i == uwatched_set_j:
                        set_map["uwatched"] = 1
                    elif uwatched_set_i * uwatched_set_j == -1:
                        set_map["uwatched"] = 0
                    # ucmt_t
                    ucmt_set_j = dev_set_map(
                        user_comment_number_list[j], ucmt_avg, ucmt_t
                    )
                    if ucmt_set_i == ucmt_set_j:
                        set_map["ucmt"] = 1
                    elif ucmt_set_i * ucmt_set_j == -1:
                        set_map["ucmt"] = 0
                    # utopic_t
                    utopic_set_j = dev_set_map(
                        user_topic_number_list[j], utopic_avg, utopic_t
                    )
                    if utopic_set_i == utopic_set_j:
                        set_map["utopic"] = 1
                    elif utopic_set_i * utopic_set_j == -1:
                        set_map["utopic"] = 0
                # add edge
                for k_a, v_a in set_map.items():
                    for k_b, v_b in set_map.items():
                        if k_a == k_b:
                            continue
                        if v_a and not v_b:
                            edge_type = k_a + "-" + k_b
                            fhandler = graph_edges.get(edge_type, None)
                            if not fhandler:
                                f = open(
                                    os.path.join(self.save_dir, edge_type + ".txt"),
                                    "w+",
                                )
                                graph_edges[edge_type] = f
                                fhandler = f
                            fhandler.write("{},{}\n".format(i, j))

    @classmethod
    def walk_dir(self, base):
        for root, ds, fs in os.walk(base):
            for f in fs:
                fullname = os.path.join(root, f)
                yield fullname

    @classmethod
    def load_graph(cls, graph_base, bidirect=True, edge_type: list = None):
        """
        :param graph_base, Graph storage location.
        """
        G = {}
        # load_graph
        for it in cls.walk_dir(graph_base):
            relation_type, filetype = it.split("/")[-1].split(".")

            if relation_type == "info" or filetype != "txt":
                continue

            if edge_type and relation_type not in edge_type:
                continue
            print(relation_type)

            df = pd.read_csv(it, header=None)
            src_list = df[0].to_list()
            dst_list = df[1].to_list()

            if bidirect:
                src_list_tmp = deepcopy(src_list)
                dst_list_tmp = deepcopy(dst_list)
                src_list.extend(dst_list_tmp)
                dst_list.extend(src_list_tmp)

            edge_index = torch.tensor([src_list, dst_list])

            relation_type = ("review", relation_type, "review")
            G[relation_type] = edge_index

            cls.relation_types.append(relation_type)

        return G

    def get_node_text_feature(self, featurizer: TextFeaturizer, node_text):
        # The BERT semantic space should utilize the same tokenizer and vocabulary numbering; it should not rely on the segmentation results from Jieba.
        # node_text_feature = featurizer.jieba_tokenize(text_data=node_text, pad=True)
        batch = 8  # TODO: Adapt the batch size according to the GPU size. Alternatively, use a try-except mechanism to continuously attempt and adjust.
        all_num = len(node_text)
        rnd = all_num // batch
        res = all_num / batch
        if res > rnd:
            rnd += 1
        batch_text_feat_list = []
        for i in range(rnd):
            # print(batch * i, batch * i + 4)
            # print(i, rnd)
            batch_text = node_text[batch * i : batch * (i + 1)]
            batch_text_tokens, batch_text_masks = featurizer.bert_tokenize(
                text_data=batch_text, pad=True
            )
            batch_text_feature = featurizer.bert_embedding(
                bert_token=batch_text_tokens, bert_mask=batch_text_masks
            )
            batch_text_feat_list.append(batch_text_feature)
        node_text_feature = np.concatenate(batch_text_feat_list, axis=0)
        return node_text_feature

    def get_node_feature(self, featurizer: TextFeaturizer):
        maoyan_comments = self.maoyan_dataset.get_maoyan_comments()
        texts = maoyan_comments.get_comment_text()
        node_text_feature = self.get_node_text_feature(
            featurizer=featurizer, node_text=texts
        )
        self.node_feature = {
            "review": node_text_feature,
        }
        return self.node_feature

    def save_node_feature(self, path=None):
        """Save the feature attributes of the graph nodes."""
        if not path:
            path = os.path.join(self.save_dir, "rgraph_node_feature")
        np.save(path, self.node_feature)
        return path

    @classmethod
    def load_node_feature(cls, path=None):
        """Load the feature attributes of the graph nodes."""
        if not path:
            path = os.path.join(cls.save_dir, "rgraph_node_feature.npy")
        node_feature = np.load(path, allow_pickle=True).item()
        return node_feature

    @classmethod
    def get_graph_meta(cls):
        """Retrieve the meta information of the heterogeneous graph."""
        meta = {"node": ["review"], "relation": cls.relation_types}
        return meta

    def save_graph_meta(self, save_path=None):
        if not save_path:
            save_path = os.path.join(self.save_dir, "rgraph_meta")
        meta = self.get_graph_meta()
        np.save(save_path, meta)
        return save_path

    @classmethod
    def load_graph_meta(cls, path=None):
        if not path:
            path = os.path.join(cls.save_dir, "rgraph_meta.npy")
        meta = np.load(path, allow_pickle=True).item()
        meta = (meta["node"], meta["relation"])
        return meta
