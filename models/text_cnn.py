import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextCNNConfig:

    def __init__(self, dataset: str, embedding: Union[str, None]) -> None:
        # model name
        self.model_name = "TextCNN"

        # data
        self.data_dir = "data/datasets"
        self.train_path = os.path.join(self.data_dir, dataset, "train.txt")
        self.dev_path = os.path.join(self.data_dir, dataset, "dev.txt")
        self.test_path = os.path.join(self.data_dir, dataset, "test.txt")
        self.class_path = os.path.join(self.data_dir, dataset, "class.txt")
        self.vocab_path = os.path.join(self.data_dir, dataset, "vocab/vocab.pkl")
        # weight
        self.save_path = os.path.join("data/weights", self.model_name + ".ckpt")
        # log
        self.log_path = os.path.join("data/log/", self.model_name)
        if not embedding or embedding == "random":
            self.embedding_pretrained = None
        else:
            self.embedding_path = os.path.join(self.data_dir, dataset, "embeddings", embedding)
            self.embedding_pretrained = torch.tensor(
                np.load(self.embedding_path)["embeddings"].astype("float32")
            )
        self.class_list = [x.strip() for x in open(self.class_path, encoding="utf-8").readlines()]

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # parameters
        self.num_embeddings = 0
        self.embedding_dim = 300
        self.pad_size = 32
        self.batch_size = 128
        self.num_epochs = 20
        self.num_filters = 256
        self.dropout = 0.5
        self.learning_rate = 1e-3
        self.filter_sizes = [2, 3, 4]
        self.num_classes = len(self.class_list)
        self.require_improvement = 1000


class TextCNN(nn.Module):
    """ 
    Convolutional Neural Networks for Sentence Classification

    https://arxiv.org/abs/1408.5882
    """

    def __init__(self, config) -> None:
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim, padding_idx=config.num_embeddings - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_dim)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
    
    def conv_and_pool(self, x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        # x: [batch_size, in_channels, seq_len, dim]
        x = F.relu(conv(x))
        # x: [batch_size, out_channels, seq_len - height + 1, width]
        # width = 1, so squeeze(3)
        x = x.squeeze(3)
        # x: [batch_size, out_channels, seq_len - height + 1]
        x = F.max_pool1d(x, x.size(2))
        # x: [batch_size, out_channels, out_pool]
        # out_pool = 1, so squeeze(2)
        x = x.squeeze(2)
        # x: [batch_size, out_channels]
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        out = self.embedding(x)
        # out: [batch_size, seq_len, dim]
        out = out.unsqueeze(1)
        # out: [batch_size, in_channels, seq_len, dim]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # out: [batch_size, out_channels * num_filters]
        out = self.dropout(out)
        out = self.fc(out)
        # out: [batch_size, num_classes]
        return out
