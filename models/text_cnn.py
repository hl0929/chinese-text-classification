import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import Config


class TextCNNConfig(Config):

    def __init__(self, dataset: str, embedding: Union[str, None]) -> None:
        super().__init__(dataset, embedding)
        # model name
        self.model_name = "TextCNN"

        # weight
        self.save_path = os.path.join("data/weights", self.model_name + ".ckpt")
        # log
        self.log_path = os.path.join("data/log/", self.model_name)

        # cnn
        self.filter_sizes = [2, 3, 4]  # convolution kernel size 
        self.num_filters = 256         # channels
        


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
            self.embedding = nn.Embedding(config.n_vocab, config.embedding_dim, padding_idx=config.n_vocab - 1)
        
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
        # x = F.max_pool1d(x, x.size(2))
        # to onnx
        x = F.max_pool1d(x, int(x.size(2)))
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
