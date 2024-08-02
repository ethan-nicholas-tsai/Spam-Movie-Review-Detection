import pandas as pd
import re
from bases.data.dataset_base import DataBase, DatasetBase


class DouBanLongComments(
    DatasetBase
):  
    """Obtain the genres of the long-reviewed movies.
    Other movies mentioned in the long review content.
    """

    def __init__(self, douban_long_comment_file):
        super(DouBanLongComments, self).__init__()
        self.douban_df = pd.read_csv(douban_long_comment_file, index_col=0)

    def get_data(self):
        return self.douban_df

    def get_data_by_id(self, douban_movie_id, comment_count=20):
        """Return the long review dataframe based on the ID.
        TODO: When it comes to external knowledge, we can only perform fuzzy searches in the form of movie names, and cannot establish a one-to-one correspondence.
        """
        douban_comments = self.douban_df.loc[self.douban_df["电影id"] == douban_movie_id]
        return douban_comments.iloc[:comment_count, :]

    def get_data_by_name(self, movie_name, comment_count=20):
        """Return the long review dataframe based on the movie title.
        TODO: There might be long reviews for movies with the same name, consider how to optimize this.
        """
        douban_comments = self.douban_df.loc[self.douban_df["电影名称"] == movie_name]
        return douban_comments.iloc[:comment_count, :]

    def get_comment_text_by_name(self, movie_name, comment_count=20):
        """Return the long review text based on the movie title.
        TODO: There might be long reviews for movies with the same name, consider how to optimize this.
        """
        douban_comments = self.douban_df.loc[self.douban_df["电影名称"] == movie_name]
        if comment_count:
            douban_comments = douban_comments.iloc[:comment_count, :]
        comment_text_list = douban_comments.评论内容.to_list()
        return comment_text_list

    def find_movie_name_in_comment_text(self, movie_comment_text):
        """Extract the movie names mentioned in the reviews.
        TODO: Optimize the pattern for precise matching (this can also be placed in the data_util module...).
        """
        pattern = re.compile("《(.*?)》")
        movie_name_list = pattern.findall(movie_comment_text)  # List of movie names in the long reviews.
        movie_name_list = list(set(movie_name_list))  # Remove duplicate mentions of the same movie.
        return movie_name_list

    def __len__(self):
        return len(self.douban_df)


class MaoYanMovies(DataBase):
    def __init__(self, movie_path):
        super(MaoYanMovies, self).__init__()
        self.movie_df = pd.read_csv(movie_path, low_memory=False)

    def get_data(self):
        return self.movie_df

    def get_data_by_name(self, movie_name):
        """Return the movie dataframe based on the movie title."""
        movie_data = self.movie_df.loc[self.movie_df["名字"] == str(movie_name)]
        return movie_data

    def get_movie_type(self):
        """Obtain a list of movie genres.
        :return movie_types
        """
        movie_types = self.movie_df.类型.tolist()
        return movie_types

    def get_movie_type_by_name(self, movie_name):
        """
        Obtain the genre of the current movie.
        :param movie_name
        :return: movie_types_list
        """
        # Get the value at a specific position.
        movie_types = self.movie_df.loc[self.movie_df["名字"] == str(movie_name)].类型
        movie_types_list = movie_types.to_list()[0].split(",")  
        return movie_types_list

    def get_movie_name(self):
        movie_names = self.movie_df.名字.tolist()
        return movie_names

    def get_movie_score(self):
        movie_scores = self.movie_df.score.tolist()
        return movie_scores

    def get_movie_publish_time(self):
        movie_publish_times = self.movie_df.上映日期.tolist()
        return movie_publish_times

    def get_movie_synopsis_by_name(self, movie_name):
        """Return a movie summary based on the movie title query."""
        movie_synopsis = self.movie_df.loc[
            self.movie_df["名字"] == movie_name
        ].简介.to_list()[0]
        return movie_synopsis

    def get_movie_publish_time_by_name(self, movie_name):
        """Return the movie release date based on the movie title query."""
        movie_publish_time = self.movie_df.loc[
            self.movie_df["名字"] == movie_name
        ].上映日期.to_list()[0]
        return movie_publish_time


class MaoYanComments(DataBase):
    def __init__(self, comment_path):
        super(MaoYanComments, self).__init__()
        print("Loading MaoYanComments...")
        self.comment_df = pd.read_excel(comment_path)

    def get_data(self):
        return self.comment_df

    def get_comment_label(self):
        """Retrieve the review tags.
        :return comment_label_list
        """
        comment_label_list = self.comment_df.标签.tolist()
        comment_label_list = [
            1 if str(label) == "1" else 0 for label in comment_label_list
        ]
        return comment_label_list

    def get_movie_name(self):
        """Retrieve all the names of the movies mentioned in the reviews.
        :return movie_name_list
        """
        movie_name_list = self.comment_df.名称.tolist()
        return movie_name_list

    def get_comment_text(self):
        """Retrieve the review text.
        :return comment_text_list
        """
        comment_text_list = self.comment_df.内容.tolist()
        return comment_text_list

    def get_comment_score(self):
        """Retrieve all the comment scores.
        :return comment_score_list
        """
        comment_score_list = self.comment_df.用户评分.to_list()  
        return comment_score_list

    def get_thumbs_up(self):
        """Retrieve all the likes.
        :return thumbs_up_list
        """
        thumbs_up_list = self.comment_df.点赞.to_list()  
        return thumbs_up_list

    def get_follow_comment_number(self):
        """Retrieve the number of people who have followed up on the review.
        :return follow_comment_number_list
        """
        follow_comment_number_list = self.comment_df.评论的评论.to_list()  
        return follow_comment_number_list

    def get_user_rank(self):
        """Retrieve the user rank.
        :return user_rank_list
        """
        user_rank_list = self.comment_df.用户等级.to_list()  
        return user_rank_list

    def get_comment_time(self):
        """Retrieve the review time
        :return comment_time_list
        """
        comment_time_list = self.comment_df.评论时间.to_list() 
        return comment_time_list

    def get_user_to_watch_number(self):
        """Retrieve the number of movies the user wants to watch.
        :return user_to_watch_number_list
        """
        user_to_watch_number_list = self.comment_df.想看.to_list() 
        return user_to_watch_number_list

    def get_user_watched_number(self):
        """Retrieve the number of movies the user has watched.
        :return user_watched_number_list
        """
        user_watched_number_list = self.comment_df.观看.to_list() 
        return user_watched_number_list

    def get_user_comment_number(self):
        """Retrieve the total number of user reviews.
        :return user_comment_number_list
        """
        user_comment_number_list = self.comment_df.用户评论总数.to_list()  
        return user_comment_number_list

    def get_user_topic_number(self):
        """Retrieve the number of topics the user is following.
        :return topic
        """
        user_topic_number_list = self.comment_df.话题.to_list()  
        return user_topic_number_list


class MaoYanDataset(DatasetBase):
    def __init__(self, comment_path="", movie_path=""):
        super(MaoYanDataset, self).__init__()
        self.maoyan_movies = MaoYanMovies(movie_path=movie_path)
        self.maoyan_comments_labeled = MaoYanComments(comment_path=comment_path)

    def get_maoyan_movies(self):
        return self.maoyan_movies

    def get_maoyan_comments(self):
        return self.maoyan_comments_labeled

    def get_movie_data(self):
        return self.maoyan_movies.get_data()

    def get_comment_data(self):
        return self.maoyan_comments_labeled.get_data()

    def get_movie_names(self, labeled=True):
        """Retrieve all movie titles.
        :param labeled
            Default is True, return the movie names in the `maoyan_comment.csv` that have been tagged.
            If False, return all movie names from the `maoyan_movie` spanning from 2017 to 2021.
        :return movie_names
        """
        if labeled:
            movie_names = self.maoyan_comments_labeled.get_movie_name()
        else:
            movie_names = self.maoyan_movies.get_movie_name()
        movie_names = list(set(movie_names))
        movie_names.sort() 
        return movie_names

    def get_movie_types(self):
        """Check how many movie genres there are in total.
        :return: maoyan_movie_types_all
        """
        movie_type_list = self.maoyan_movies.get_movie_type()
        maoyan_movie_types_all = set()

        for movie_type in movie_type_list:
            try:
                x = movie_type.split(",")
                maoyan_movie_types_all.update(x)  # Add all the new `movie_type` entries.
            except:
                continue
        maoyan_movie_types_all = list(maoyan_movie_types_all)

        return maoyan_movie_types_all

    def get_movie_type_by_name(self, movie_name):
        """Retrieve the current movie genre based on the movie title.
        :param movie_name
        :return: movie_type
        """
        movie_type = self.maoyan_movies.get_movie_type_by_name(movie_name=movie_name)
        return movie_type

    def get_movie_synopsis_by_name(self, movie_name):
        """Retrieve the current movie synopsis based on the movie title.
        :param movie_name
        :return movie_synopsis
        """
        movie_synopsis = self.maoyan_movies.get_movie_synopsis_by_name(
            movie_name=movie_name
        )
        return movie_synopsis

    def get_movie_publish_time_by_name(self, movie_name):
        """Retrieve the release date of the current movie based on the movie title.
        :param movie_name
        :return movie_publish_time
        """
        movie_publish_time = self.maoyan_movies.get_movie_publish_time_by_name(
            movie_name=movie_name
        )
        return movie_publish_time
