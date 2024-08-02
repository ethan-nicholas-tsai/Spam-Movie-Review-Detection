import pandas as pd
import numpy as np
from bases.data.transform_base import FeaturizerBase
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils.data_util import CJiebaTokenizer, CBertTokenizer
import torch


class TextFeaturizer(FeaturizerBase):
    def __init__(
        self,
        bert_vocab_path=None,
        jieba_vocab_path=None,
        max_seq_len=None,
        bert_model_path=None,
        device="cpu",
    ):
        if bert_vocab_path:
            self.load_bert_tokenizer(bert_vocab_path)
        if jieba_vocab_path:
            self.load_jieba_tokenizer(jieba_vocab_path)
        self.max_seq_len = max_seq_len
        self.bert = None
        # 设备参数
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if bert_model_path:
            self.load_bert_model(bert_model_path)

    def load_bert_model(self, bert_model_path):
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.bert.to(self.device)
        return self.bert

    def set_max_seq_len(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def load_bert_tokenizer(self, vocab_path=None):
        self.bert_tokenizer = CBertTokenizer(vocab_path=vocab_path)
        return self.bert_tokenizer

    def load_jieba_tokenizer(self, vocab_path=None):
        self.jieba_tokenizer = CJiebaTokenizer(vocab_path=vocab_path)
        return self.jieba_tokenizer

    def bert_tokenize(self, text_data: list, pad=False, max_len=None):
        if not max_len:
            max_len = self.max_seq_len
        tokenized_text_list = []
        tokenized_text_mask_list = []
        for text in text_data:
            seq, seq_mask = self.bert_tokenizer.tokenize(
                str(text), pad=pad, max_len=max_len
            )
            tokenized_text_list.append(seq)
            tokenized_text_mask_list.append(seq_mask)
        return tokenized_text_list, tokenized_text_mask_list

    def jieba_tokenize(self, text_data: list, pad=False, max_len=None):
        if not max_len:
            max_len = self.max_seq_len
        tokenized_text_list = []
        for text in text_data:
            seq = self.jieba_tokenizer.tokenize(str(text), pad=pad, max_len=max_len)
            tokenized_text_list.append(seq)
        return tokenized_text_list

    def bert_embedding(self, bert_token, bert_mask):
        if not self.bert:
            print("The pre-trained BERT model has not been loaded.")
            return None
        bert_token = torch.tensor(bert_token).to(self.device)
        bert_mask = torch.tensor(bert_mask).to(self.device)
        word_embedding, sentence_embedding = self.bert(
            bert_token, attention_mask=bert_mask, output_all_encoded_layers=False
        )
        if self.device.type == "cpu":
            return sentence_embedding.numpy()
        elif self.device.type == "cuda":
            return sentence_embedding.cpu().detach().numpy()
        return sentence_embedding


if __name__ == "__main__":
    pass
