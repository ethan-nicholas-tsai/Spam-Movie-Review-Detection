import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, LineByLineTextDataset
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


def finetune():
    # device = "cuda:1"

    # device = torch.device(device if torch.cuda.is_available() else "cpu")

    config = BertConfig(
        vocab_size=21128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512,
    )
    # NOTE: Firstly, rename the vocabulary file in the `bert_pretrain` folder to `vocab.txt`, and the configuration file to `config.json`.
    tokenizer = BertTokenizer.from_pretrained("models/pretrained/BERT/bert_pretrain")
    model = BertForMaskedLM(config).from_pretrained(
        "models/pretrained/BERT/bert_pretrain"
    )
    print("No of parameters: ", model.num_parameters())

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    from transformers import Trainer, TrainingArguments

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="playground/text.txt",  # Corpus to train
        block_size=256,  # 64  # maximum sequence length
    )

    print("No. of lines: ", len(dataset))  # No of lines in your datset

    # model.to(device)
    # dataset.to(device)

    training_args = TrainingArguments(
        output_dir="models/pretrained/BERT",
        overwrite_output_dir=True,
        num_train_epochs=30,  # 30,
        per_device_train_batch_size=32,  # 64,
        save_steps=10000,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model("models/pretrained/BERT/bert_finetune")


if __name__ == "__main__":
    pass
