"""
Preprocess the data (Segementation)
"""
import os
import pandas as pd
import jieba
from pytorch_pretrained_bert import BertTokenizer
import random
import numpy as np

UNK, PAD = "<UNK>", "<PAD>"  # Unknown and Padding token


class BaseTokenizer:
    def __init__(self):
        pass

    @classmethod
    def load_vocab(cls, vocab_path):
        with open(vocab_path, "r") as f:
            vocab = [line.strip("\n") for line in f.readlines()]
        return vocab

    @classmethod
    def get_vocab_size(cls, vocab_path):
        vocab = cls.load_vocab(vocab_path=vocab_path)
        vocab_size = len(vocab) if vocab[-1] == PAD else len(vocab) + 2
        return vocab_size


class CharTokenizer:
    def __init__(self, vocab_path=None):
        """
        :param vocab_path: vocabulary list file path
        """
        self.tokenizer = lambda x: [y for y in x]  # char-level
        if vocab_path:
            if ".txt" in vocab_path:
                with open(vocab_path, "r") as f:
                    self.vocab = [line.strip("\n") for line in f.readlines()]
                    self.word2ids = {word: idx for idx, word in enumerate(self.vocab)}
            self.word2ids.update({UNK: len(self.word2ids), PAD: len(self.word2ids) + 1})

    def tokenize(self, seq, pad=False, max_len=256):
        """
        Return token list
        :param seq: sequence of tokens
        :return: list
        """
        tokens = self.tokenizer(seq)
        token_list = []
        for token in tokens:
            token_list.append(self.word2ids.get(token, self.word2ids.get(UNK)))

        if pad:
            token_list = self.pad(token_list=token_list, max_len=max_len)
        return token_list

    def pad(self, token_list, max_len=256):
        """
        :param max_len: maximal sequence length (truncate and padding the sequence accordingly)
        """
        pad = lambda x: x + (max_len - len(x)) * [self.word2ids[PAD]]  # padding function
        if len(token_list) < max_len:
            token_list = pad(token_list)
        else:
            token_list = token_list[:max_len]
        return token_list

    def build_vocab(self, data, min_freq=1, stopword=False):
        """Build vocabulary list
        :param min_freq: minimal frequency of word
        :param data: list of text
        """
        vocab_dic = {}
        stopwords_set = set([])
        if stopword:
            stopwords = (
                """~!@#$%^&*()_+`1234567890-={}[]:：";'<>,.?/|\、·！（）￥“”‘’《》，。？/—-【】…."""
            )
            stopwords_set = set([i for i in stopwords])
            stopwords_set.add("br")  # Add OOVs for removal

        for text in data:
            if not text or type(text) != str:
                continue
            for s in stopwords_set:
                text = text.strip().replace(s, "")
                text = text.replace("   ", " ").replace("  ", " ")
                text = text.replace("   ", " ").replace("  ", " ")
            for word in self.tokenizer(text):
                if word.strip():
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted(
            [_ for _ in vocab_dic.items() if _[1] >= min_freq],
            key=lambda x: x[1],
            reverse=True,
        )
        self.vocab = [it[0] for it in vocab_list]
        self.word2ids = {
            word_count[0]: idx for idx, word_count in enumerate(vocab_list)
        }
        self.word2ids.update(
            {UNK: len(vocab_dic), PAD: len(vocab_dic) + 1}
        )  # Unknown and Padding token
        return vocab_list

    def write_vocab(self, save_dir):
        with open(os.path.join(save_dir, "vocab.txt"), "w") as f:
            for word in self.vocab:
                f.write(word)
                f.write("\n")


class CJiebaTokenizer:
    def __init__(self, vocab_path=None):
        """
        Tokenizer
        :param vocab_path: vocabulary list
        """
        if vocab_path:
            if ".csv" in vocab_path:
                df = pd.read_csv(vocab_path)
                self.word2ids = {
                    y: x for x, y in zip(df.index.to_list(), df.分词.to_list())
                }
            elif ".txt" in vocab_path:
                with open(vocab_path, "r") as f:
                    self.vocab = [line.strip("\n") for line in f.readlines()]
                    self.word2ids = {word: idx for idx, word in enumerate(self.vocab)}
            self.word2ids.update({UNK: len(self.word2ids), PAD: len(self.word2ids) + 1})

    def tokenize(self, seq, pad=False, max_len=256):
        """
        Return token list
        :param seq: sequence of token
        :return: list
        """
        tokens = jieba.cut(seq)
        token_list = []
        for token in tokens:
            token_list.append(self.word2ids.get(token, self.word2ids.get(UNK)))

        if pad:
            token_list = self.pad(token_list=token_list, max_len=max_len)
        return token_list

    def pad(self, token_list, max_len=256):
        """
        :param max_len: maximal sequence length (truncate and padding the sequence accordingly)
        """
        pad = lambda x: x + (max_len - len(x)) * [self.word2ids[PAD]]  # padding function
        if len(token_list) < max_len:
            token_list = pad(token_list)
        else:
            token_list = token_list[:max_len]
        return token_list

    def build_vocab(self, data, min_freq=1, stopword=False):
        """Build vocabulary list
        :param min_freq: minimal frequency of word
        :param data: list of text
        """
        vocab_dic = {}
        stopwords_set = set([])
        if stopword:
            stopwords = (
                """~!@#$%^&*()_+`1234567890-={}[]:：";'<>,.?/|\、·！（）￥“”‘’《》，。？/—-【】…."""
            )
            stopwords_set = set([i for i in stopwords])
            stopwords_set.add("br")  # Add OOVs for removal

        for text in data:
            if not text or type(text) != str:
                continue
            for s in stopwords_set:
                text = text.strip().replace(s, "")
                text = text.replace("   ", " ").replace("  ", " ")
                text = text.replace("   ", " ").replace("  ", " ")
            for word in jieba.cut(text):
                if word.strip():
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted(
            [_ for _ in vocab_dic.items() if _[1] >= min_freq],
            key=lambda x: x[1],
            reverse=True,
        )
        self.vocab = [it[0] for it in vocab_list]
        self.word2ids = {
            word_count[0]: idx for idx, word_count in enumerate(vocab_list)
        }
        self.word2ids.update(
            {UNK: len(vocab_dic), PAD: len(vocab_dic) + 1}
        )  # Unknown and Padding token
        return vocab_list

    def write_vocab(self, save_dir):
        with open(os.path.join(save_dir, "vocab.txt"), "w") as f:
            for word in self.vocab:
                f.write(word)
                f.write("\n")


class CBertTokenizer:
    def __init__(self, vocab_path=None):
        self.tokenizer = BertTokenizer(vocab_path)

    def tokenize(self, text, pad=False, max_len=512):
        seq = self.tokenizer.tokenize(str(text).strip())
        seq_mask = None
        if pad:
            seq, seq_mask = self.pad(seq, max_seq_len=max_len)
        return seq, seq_mask

    def segment_text(self, text, seg_len=512):
        """Segement the text into multi pieces"""
        # seq = self.tokenizer.tokenize(str(text).strip())
        seq = text.strip()
        seg_text_list = []
        all_len = len(seq)
        seg_len = seg_len - 2
        segs = all_len // seg_len
        segs = segs + 1 if segs < all_len / seg_len else segs
        for i in range(segs):
            seg = seq[i * seg_len : (i + 1) * seg_len]
            # seg_text_list.append("".join(seg))
            seg_text_list.append(seg)
        return seg_text_list

    def pad(self, seq, max_seq_len=512):
        """
        1. Since this class deals with single-sentence sequences, following the sequence processing method in BERT, special characters 'CLS' and 'SEP' need to be concatenated at the beginning and end of the input sequence respectively. Therefore, the sequence length without the two special characters should be less than or equal to max_seq_len - 2. If the sequence length exceeds this value, it needs to be truncated.
        2. The input sequence is ultimately formed into a sequence of ['CLS', seq, 'SEP']. If the length of this sequence is less than max_seq_len, it is padded with zeros.

        Input:
            seq         : The input sequence (sentence)
            max_seq_len : The length of the sequence after concatenating the special characters 'CLS' and 'SEP'.

        Output:
            seq         : The sequence has had the 'CLS' and 'SEP' tokens concatenated at the beginning and end of the input parameter 'seq', respectively. If the resulting length is still less than 'max_seq_len', it has been padded with zeros at the end.
            seq_mask    : A sequence containing only 0s and 1s, with a length equal to that of 'seq', is used to represent whether the symbols in 'seq' are meaningful. If the corresponding position in the 'seq' sequence is a padding token, then the value is 1; otherwise, it is 0.
        """
        # Truncate the overly long sequence.
        if len(seq) > (max_seq_len - 2):
            seq = seq[0 : (max_seq_len - 2)]
        # Concatenate special symbols at the beginning and end respectively.
        seq = ["[CLS]"] + seq + ["[SEP]"]
        # Token to id
        seq = self.tokenizer.convert_tokens_to_ids(seq)
        # Generate a padding sequence based on the lengths of `max_seq_len` and `seq`.
        padding = [0] * (max_seq_len - len(seq))
        # Create seq_mask
        seq_mask = [1] * len(seq) + padding
        # Append a padding sequence to `seq`.
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        return seq, seq_mask


class DataSampler:
    def __init__(self):
        pass

    @classmethod
    def bi_cls_data_sampler_evaluator(
        cls, labels: list, cls_max_ratio=0.5, num_base=1000
    ):
        """Evaluator for the binary classification imbalance sampling method
        Assessing the maximum experimental dataset size that can be constructed given the specified ratios of positive and negative samples and the original dataset's sample distribution.
        :param labels, label list
        :cls_max_ratio, maximal label ratio
        :num_base, minimal batch
        :return max_sample_num
        """
        print(len(labels))
        from collections import Counter

        cls_distribution = Counter(labels)
        print(cls_distribution)
        cls_list = list(cls_distribution.keys())
        min_cls_num = cls_distribution[cls_list[0]]
        min_cls = cls_list[0]
        for cls in cls_list[1:]:
            cls_num = cls_distribution[cls]
            if cls_num < min_cls_num:
                min_cls = cls
                min_cls_num = cls_num
        print(min_cls, min_cls_num)
        max_sample_num = min_cls_num // cls_max_ratio
        max_sample_num = int(max_sample_num // num_base * num_base)
        print(max_sample_num)
        return max_sample_num

    @classmethod
    def bi_cls_data_sampler(cls, data: list, dataset_scale, pos_ratio=0.5, seed=2022):
        """Binary classification imbalance sampling method
        :param data, the first elements are labels
        :return sample_data
        """
        labels = data[0]
        pos_sample_num = int(dataset_scale * pos_ratio)
        neg_sample_num = int(dataset_scale - pos_sample_num)

        label_index_list = [(labels[i], i) for i in range(len(labels))]

        random.seed(seed)
        random.shuffle(label_index_list)

        pos_index = []
        neg_index = []

        for i in range(len(labels)):
            label_index = label_index_list[i]
            label = label_index[0]
            index = label_index[1]
            if label:
                pos_index.append(index)
            else:
                neg_index.append(index)

        pos_index = pos_index[:pos_sample_num]
        neg_index = neg_index[:neg_sample_num]
        all_index = pos_index + neg_index

        random.seed(seed * 3)
        random.shuffle(all_index)

        sample_data = []
        for datum in data:
            datum = np.array(datum)
            sample_datum = datum[all_index].tolist()
            sample_data.append(sample_datum)
        return sample_data

    @classmethod
    def bi_cls_data_pipe(
        cls, data: list, pos_ratio, num_base, seed, dataset_scale=None
    ):
        labels = data[0]
        if not dataset_scale:
            dataset_scale = cls.bi_cls_data_sampler_evaluator(
                labels, cls_max_ratio=pos_ratio, num_base=num_base
            )
        print(dataset_scale)
        dataset = cls.bi_cls_data_sampler(
            data=data, dataset_scale=dataset_scale, pos_ratio=pos_ratio, seed=seed
        )
        return dataset


class NewDataSampler:
    def __init__(self):
        pass

    @classmethod
    def bi_cls_data_sampler_evaluator(
        cls, labels: list, cls_max_ratio=0.5, num_base=1000
    ):
        """Evaluator for the binary classification imbalance sampling method
        Assessing the maximum experimental dataset size that can be constructed given the specified ratios of positive and negative samples and the original dataset's sample distribution.
        :param labels, label list
        :cls_max_ratio, maximal label ratio
        :num_base, minimal batch
        :return max_sample_num
        """
        print(len(labels))
        from collections import Counter

        cls_distribution = Counter(labels)
        print(cls_distribution)
        cls_list = list(cls_distribution.keys())
        min_cls_num = cls_distribution[cls_list[0]]
        min_cls = cls_list[0]
        for cls in cls_list[1:]:
            cls_num = cls_distribution[cls]
            if cls_num < min_cls_num:
                min_cls = cls
                min_cls_num = cls_num
        print(min_cls, min_cls_num)
        max_sample_num = min_cls_num // cls_max_ratio
        max_sample_num = int(max_sample_num // num_base * num_base)
        print(max_sample_num)
        return max_sample_num

    @classmethod
    def bi_cls_data_sampler(cls, data: list, dataset_scale, pos_ratio=0.5, seed=2022):
        """Binary classification imbalance sampling method
        :param data, the first elements are labels
        :return sample_data
        """
        labels = data[0]
        pos_sample_num = int(dataset_scale * pos_ratio)
        neg_sample_num = int(dataset_scale - pos_sample_num)

        label_index_list = [(labels[i], i) for i in range(len(labels))]

        random.seed(seed)
        random.shuffle(label_index_list)

        pos_index = []
        neg_index = []

        for i in range(len(labels)):
            label_index = label_index_list[i]
            label = label_index[0]
            index = label_index[1]
            if label:
                pos_index.append(index)
            else:
                neg_index.append(index)

        pos_index = pos_index[:pos_sample_num]
        neg_index = neg_index[:neg_sample_num]
        all_index = pos_index + neg_index

        random.seed(seed * 3)
        random.shuffle(all_index)

        sample_data = []
        for datum in data:
            datum = np.array(datum)
            sample_datum = datum[all_index].tolist()
            sample_data.append(sample_datum)
        return sample_data, all_index

    @classmethod
    def bi_cls_data_pipe(
        cls, data: list, pos_ratio, num_base, seed, dataset_scale=None
    ):
        labels = data[0]
        if not dataset_scale:
            dataset_scale = cls.bi_cls_data_sampler_evaluator(
                labels, cls_max_ratio=pos_ratio, num_base=num_base
            )
        print(dataset_scale)
        dataset, index = cls.bi_cls_data_sampler(
            data=data, dataset_scale=dataset_scale, pos_ratio=pos_ratio, seed=seed
        )
        return dataset, index


if __name__ == "__main__":
    pass
