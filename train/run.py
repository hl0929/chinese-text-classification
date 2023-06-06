import sys
sys.path.append(".")

import torch
import argparse

import numpy as np
from utils.utils import build_dataset, DatasetIterater
from scripts import init_network, train
from models import TextCNN, TextCNNConfig


def set_seed():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True


def main():
    set_seed()
    
    parser = argparse.ArgumentParser(description="Text Classification")
    parser.add_argument("--model", type=str, required=True, help="Support: TextCNN | TextRNN")
    parser.add_argument("--embedding", default="pre-trained", type=str, help="Support: random | pre-trained")
    parser.add_argument("--token-level", default="char", type=str, help="Support token level: char or word")
    parser.add_argument("--dataset", default="THUCNews", type=str, help="Support datasets: THUCnews")
    args = parser.parse_args()

    # embedding
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == "random":
        embedding = "random"
    print(embedding)
    
    # config
    model_name = args.model
    if model_name == "TextCNN":
        config = TextCNNConfig(args.dataset, embedding)

    # dataset
    vocab, train_data, dev_data, test_data = build_dataset(config, args.token_level)
    config.num_embeddings = len(vocab)
    train_iter = DatasetIterater(train_data, config)
    dev_iter = DatasetIterater(dev_data, config)
    test_iter = DatasetIterater(test_data, config)
    
    # model
    if model_name == "TextCNN":
        model = TextCNN(config).to(config.device)

    # network initialization
    init_network(model=model)

    # train
    train(config, model, train_iter, dev_iter, test_iter)


if __name__ == "__main__":
    main()