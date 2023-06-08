import os
from typing import Union

import torch
import torch.nn as nn
from models.config import Config


class TextRNNConfig(Config):

    def __init__(self, dataset: str, embedding: Union[str, None]) -> None:
        super().__init__(dataset, embedding)
        # model name
        self.model_name = "TextRNN"

        # weight
        self.save_path = os.path.join("data/weights", self.model_name + ".ckpt")
        # log
        self.log_path = os.path.join("data/log/", self.model_name)

        # rnn
        self.hidden_size = 128  # lstm hidden size         
        self.num_layers = 2     # lstm layer


class TextRNN(nn.Module):
    """ 
    Recurrent Neural Network for Text Classification with Multi-Task Learning

    https://arxiv.org/abs/1605.05101
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
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        out = self.embedding(x)
        # out: [batch_size, seq_len, dim]
        out, _ = self.bilstm(out)
        # out: [batch_size, seq_len, hidden_size * 2]
        out = out[:, -1, :]
        # out: [batch_size, hidden_size * 2]
        out = self.fc(out)
        # out: [batch_size, num_classes]
        return out
    