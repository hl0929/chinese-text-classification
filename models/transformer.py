import os
import copy
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.config import Config


class TransformerConfig(Config):

    def __init__(self, dataset: str, embedding: Union[str, None]) -> None:
        super().__init__(dataset, embedding)
        # model name
        self.model_name = "Transformer"

        # weight
        self.save_path = os.path.join("data/weights", self.model_name + ".ckpt")
        # log
        self.log_path = os.path.join("data/log/", self.model_name)

        # parameter
        self.model_dim = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2

        self.require_improvement = 2000   
        self.learning_rate = 5e-4  


class Transformer(nn.Module):
    """ 
    Attention Is All You Need
    
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embedding_dim, padding_idx=config.n_vocab - 1)

        self.position_embedding = Positional_Encoding(config.embedding_dim, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(config.model_dim, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList(
            [copy.deepcopy(self.encoder) for _ in range(config.num_encoder)]
        )
        self.fc = nn.Linear(config.pad_size * config.model_dim, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = self.position_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), - 1)
        out = self.fc(out)
        return out


class Encoder(nn.Module):

    def __init__(self, model_dim, num_head, hidden, dropout):
        super().__init__()
        self.attention = Multi_Head_Attention(model_dim, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(model_dim, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out
    

class Multi_Head_Attention(nn.Module):

    def __init__(self, model_dim, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert model_dim % num_head == 0
        self.dim_head = model_dim // self.num_head
        self.fc_Q = nn.Linear(model_dim, num_head * self.dim_head)
        self.fc_K = nn.Linear(model_dim, num_head * self.dim_head)
        self.fc_V = nn.Linear(model_dim, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5  
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out
    

class Positional_Encoding(nn.Module):

    def __init__(self, embed, pad_size, dropout, device):
        super().__init__()
        self.device = device
        self.pe = torch.tensor(
            [
                [pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)
            ]
        )
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x.to(self.device) + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out
    

class Scaled_Dot_Product_Attention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context
    

class Multi_Head_Attention(nn.Module):

    def __init__(self, model_dim, num_head, dropout=0.0):
        super().__init__()
        self.num_head = num_head
        assert model_dim % num_head == 0
        self.dim_head = model_dim // self.num_head
        self.fc_Q = nn.Linear(model_dim, num_head * self.dim_head)
        self.fc_K = nn.Linear(model_dim, num_head * self.dim_head)
        self.fc_V = nn.Linear(model_dim, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5 
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x 
        out = self.layer_norm(out)
        return out
    

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, model_dim, hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, hidden)
        self.fc2 = nn.Linear(hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x 
        out = self.layer_norm(out)
        return out
