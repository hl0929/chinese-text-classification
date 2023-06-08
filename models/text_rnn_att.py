import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import Config


class TextRNNAttConfig(Config):

    def __init__(self, dataset: str, embedding: Union[str, None]) -> None:
        super().__init__(dataset, embedding)
        # model name
        self.model_name = "TextRNNAtt"

        # weight
        self.save_path = os.path.join("data/weights", self.model_name + ".ckpt")
        # log
        self.log_path = os.path.join("data/log/", self.model_name)

        # rnn
        self.hidden_size = 128  # lstm hidden size         
        self.num_layers = 2     # lstm layer
        self.hidden_size2 = 64  # fc hidden size


class TextRNNAtt(nn.Module):
    """ 
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification

    https://paperswithcode.com/paper/attention-based-bidirectional-long-short-term
    """
    def __init__(self, config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embedding_dim, padding_idx=config.n_vocab - 1)
        
        self.bilstm = nn.LSTM(
            config.embedding_dim, config.hidden_size, config.num_layers,
            bidirectional=True, batch_first=True, dropout=config.dropout
        )
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc2 = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed: [batch_size, seq_len, dim]
        H, _ = self.bilstm(embed)
        # H: [batch_size, seq_len, hidden_size * 2]

        M = self.tanh(H)
        # M: [batch_size, seq_len, hidden_size * 2]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1)
        # alpha: [batch_size, seq_len]
        alpha = alpha.unsqueeze(-1)
        # alpha: [batch_size, seq_len, 1]
        out = H * alpha
        # out: [batch_size, seq_len, hidden_size * 2]

        out = torch.sum(out, 1)
        # out: [batch_size, hidden_size * 2]
        out = F.relu(out)
        out = self.fc1(out)
        # out: [batch_size, hidden_size2]
        out = self.fc2(out)
        # out: [batch_size, num_classes]
        return out