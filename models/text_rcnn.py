import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import Config


class TextRCNNConfig(Config):

    def __init__(self, dataset: str, embedding: Union[str, None]) -> None:
        super().__init__(dataset, embedding)
        # model name
        self.model_name = "TextRCNN"

        # weight
        self.save_path = os.path.join("data/weights", self.model_name + ".ckpt")
        # log
        self.log_path = os.path.join("data/log/", self.model_name)

        # rnn
        self.hidden_size = 256  # lstm hidden size         
        self.num_layers = 1     # lstm layer


class TextRCNN(nn.Module):
    """ 
    Recurrent Convolutional Neural Networks for Text Classification
    
    https://paperswithcode.com/paper/recurrent-convolutional-neural-networks-for-2
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
        self.max_pool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embedding_dim, config.num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed: [batch_size, seq_len, dim]
        H, _ = self.bilstm(embed)
        # H: [batch_size, seq_len, hidden_size * 2]

        out = torch.cat((embed, H), 2)
        # out: [batch_size, seq_len, dim + hidden_size * 2]
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        # out: [batch_size, dim + hidden_size * 2, seq_len]
        out = self.max_pool(out)
        # out: [batch_size, dim + hidden_size * 2, 1]
        out = out.squeeze()
        # out: [batch_size, dim + hidden_size * 2]
        out = self.fc(out)
        # out: [batch_size, num_classes]
        return out