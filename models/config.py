import os
from typing import Union

import torch
import numpy as np


class Config:

    def __init__(self, dataset: str, embedding: Union[str, None]) -> None:
        # data
        self.data_dir = "data/datasets"
        self.train_path = os.path.join(self.data_dir, dataset, "train.txt")
        self.dev_path = os.path.join(self.data_dir, dataset, "dev.txt")
        self.test_path = os.path.join(self.data_dir, dataset, "test.txt")
        self.class_path = os.path.join(self.data_dir, dataset, "class.txt")
        self.vocab_path = os.path.join(self.data_dir, dataset, "vocab/vocab.pkl")

        if not embedding or embedding == "random":
            self.embedding_pretrained = None
        else:
            self.embedding_path = os.path.join(self.data_dir, dataset, "embeddings", embedding)
            self.embedding_pretrained = torch.tensor(
                np.load(self.embedding_path)["embeddings"].astype("float32")
            )
        self.class_list = [x.strip() for x in open(self.class_path, encoding="utf-8").readlines()]
        self.num_classes = len(self.class_list)
        
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # paramters
        self.n_vocab = 0
        self.embedding_dim = 300
        self.pad_size = 32
        self.batch_size = 128
        self.num_epochs = 20
        self.dropout = 0.5
        self.learning_rate = 1e-3
        self.require_improvement = 1000