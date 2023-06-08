import sys
sys.path.append(".")

import torch
import unittest
from models import TextRNNConfig, TextRNN


class TestTextCNN(unittest.TestCase):

    def test_forward_random(self):
        # setup
        embedding = "random"
        dataset = "THUCNews"
        config = TextRNNConfig(embedding=embedding, dataset=dataset)
        config.n_vocab = 10  # vocab size
        model = TextRNN(config=config)

        # action
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8]])
        outputs = model(inputs)

        # assert
        self.assertEqual(outputs.size(0), 3)
        self.assertEqual(outputs.size(1), 10)

    def test_forward_pretrained(self):
        # setup
        embedding = "embedding_SougouNews.npz"
        dataset = "THUCNews"
        config = TextRNNConfig(embedding=embedding, dataset=dataset)
        model = TextRNN(config=config)

        # action
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8]])
        outputs = model(inputs)

        # assert
        self.assertEqual(outputs.size(0), 3)
        self.assertEqual(outputs.size(1), 10)

