import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import Config


class FastTextConfig(Config):

    def __init__(self, dataset: str, embedding: Union[str, None]) -> None:
        super().__init__(dataset, embedding)
        # model name
        self.model_name = "FastText"

        # weight
        self.save_path = os.path.join("data/weights", self.model_name + ".ckpt")
        # log
        self.log_path = os.path.join("data/log/", self.model_name)

        # cnn
        self.hidden_size = 256     # hidden size
        self.ngram_vocab = 250499  # ngram vocab size


class FastText(nn.Module):
    """ 
    Bag of Tricks for Efficient Text Classification
    
    https://arxiv.org/abs/1607.01759
    """
    def __init__(self, config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embedding_dim, padding_idx=config.n_vocab - 1)
        
        self.embedding_bigram = nn.Embedding(config.ngram_vocab, config.embedding_dim)
        self.embedding_trigram = nn.Embedding(config.ngram_vocab, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(config.embedding_dim * 3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        unigram = self.embedding(x[0])
        # unigram: [batch_size, seq_len, dim]
        bigram = self.embedding_bigram(x[1])
        # bigram: [batch_size, seq_len, dim]
        trigram = self.embedding_trigram(x[2])
        # trigram: [batch_size, seq_len, dim]
        out = torch.cat((unigram, bigram, trigram), -1)
        # out: [batch_size, seq_len, dim * 3]
        out = out.mean(dim=1)
        # out: [batch_size, dim * 3]
        out = self.dropout(out)
        out = self.fc1(out)
        # out: [batch_size, hidden_size]
        out = self.relu(out)
        out = self.fc2(out)
        # out: [batch_size, num_classes]
        return out
