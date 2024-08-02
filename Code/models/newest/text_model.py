from torch_geometric.nn import Linear
from bases.model_base import Network
from pytorch_pretrained_bert import BertModel


class TextBModel(Network):
    def __init__(
        self,
        out_dim,
        bert_path,
        bert_finetune=False,
        finetune_layers=["word_embeddings", "position_embeddings", "LayerNorm"],
        device=None,
    ):
        super(TextBModel, self).__init__(device=device)
        # Text feature extraction model
        self.bert = BertModel.from_pretrained(bert_path)
        # Freeze the parameters
        for param_name, param in self.bert.named_parameters():
            param.requires_grad = bert_finetune
            for it in finetune_layers:
                if it in param_name:
                    param.requires_grad = True
                    break
        self.alinear = Linear(-1, out_dim)


    def forward(self, X, mask):
        word_embedding, sentence_embedding = self.bert(
            X, attention_mask=mask, output_all_encoded_layers=False
        )
        text_feature = self.alinear(sentence_embedding)
        return text_feature
