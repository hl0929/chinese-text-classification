import sys
sys.path.append(".")

import torch
import unittest
from models import TransformerConfig, Transformer


class TestTextCNN(unittest.TestCase):

    def test_forward_random(self):
        # setup
        embedding = "random"
        dataset = "THUCNews"
        config = TransformerConfig(embedding=embedding, dataset=dataset)
        config.pad_size = 4
        config.n_vocab = 10  # vocab size
        model = Transformer(config=config).to(config.device)

        # action
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8]]).to(config.device)
        outputs = model(inputs)

        # assert
        self.assertEqual(outputs.size(0), 3)
        self.assertEqual(outputs.size(1), 10)

    def test_forward_pretrained(self):
        # setup
        embedding = "embedding_SougouNews.npz"
        dataset = "THUCNews"
        config = TransformerConfig(embedding=embedding, dataset=dataset)
        config.pad_size = 4
        model = Transformer(config=config).to(config.device)

        # action
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8]]).to(config.device)
        outputs = model(inputs)

        # assert
        self.assertEqual(outputs.size(0), 3)
        self.assertEqual(outputs.size(1), 10)

