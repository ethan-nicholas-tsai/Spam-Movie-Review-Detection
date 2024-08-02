

class FeaturizerBase(object):  
    def __init__(
        self,
        # enable_tokenizer=False,
        # enable_bert=False,
        # tokenizer_path=None,
        # bert_path=None,
    ):
        """TODO: add param enable_pretrained_tokenizer, enable_pretrained_bert..."""
        super(FeaturizerBase, self).__init__()

    def normalize(self, numeric_data: list, max_thresh=None, min_thresh=None):
        """归一化"""
        numeric_data = [float(it) for it in numeric_data]
        min_ = min(numeric_data)
        max_ = max(numeric_data)
        if min_thresh:
            min_ = min_thresh if min_ < min_thresh else min_
        if max_thresh:
            max_ = max_thresh if max_ > max_thresh else max_
        numeric_data = [(i - min_) / (max_ - min_) for i in numeric_data]
        return numeric_data

    def categorize(self, categorical_data: list):
        pass

    def standardize(self, numeric_data: list):
        pass
