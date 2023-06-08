import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import Config


class DPCNNConfig(Config):

    def __init__(self, dataset: str, embedding: Union[str, None]) -> None:
        super().__init__(dataset, embedding)
        # model name
        self.model_name = "DPCNN"

        # weight
        self.save_path = os.path.join("data/weights", self.model_name + ".ckpt")
        # log
        self.log_path = os.path.join("data/log/", self.model_name)

        # rnn
        self.num_filters = 250  # channels


class DPCNN(nn.Module):
    """ 
    Deep Pyramid Convolutional Neural Networks for Text Categorization

    https://paperswithcode.com/paper/deep-pyramid-convolutional-neural-networks
    """
    def __init__(self, config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embedding_dim, padding_idx=config.n_vocab - 1)

        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_dim), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed: [batch_size, seq_len, dim]
        out = embed.unsqueeze(1)
        # out: [batch_size, 1, seq_len, dim]
        out = self.conv_region(out)
        # out: [batch_size, num_filters, seq_len-3+1, 1]

        out = self.padding1(out)
        # out: [batch_size, num_filters, seq_len(seq_len-3+1+2), 1]
        out = self.relu(out)
        out = self.conv(out)
        # out: [batch_size, num_filters, seq_len-3+1, 1]

        out = self.padding1(out)
        # out: [batch_size, num_filters, seq_len(seq_len-3+1+2), 1]
        out = self.relu(out)
        out = self.conv(out)
        # out: [batch_size, num_filters, seq_len-3+1, 1]
        while out.size()[2] >= 2:
            out = self._block(out)
        out = out.squeeze()
        x = self.fc(out)
        return x

    def _block(self, x):
        # set seq_len = 4
        # x: [batch_size, num_filters, seq_len-3+1, 1]
        # x: [batch_size, num_filters, 2, 1]
        x = self.padding2(x)
        # x: [batch_size, num_filters, seq_len-3+1+1, 1]
        # x: [batch_size, num_filters, 3, 1]
        px = self.max_pool(x)
        # x: [batch_size, num_filters, (seq_len-4)/2 + 1 = (seq_len-3+1+1 -2-1)/2 + 1, 1]
        # x: [batch_size, num_filters, 1, 1]

        x = self.padding1(px)
        # x: [batch_size, num_filters, (seq_len-4)/2 + 1 +1+1, 1]
        # x: [batch_size, num_filters, 3, 1]
        x = self.relu(x)
        x = self.conv(x)
        # x: [batch_size, num_filters, (3-2-1)/2+1, 1]
        # x: [batch_size, num_filters, 1, 1]
        

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)

        x = x + px
        return x
