import sys
sys.path.append(".")

import os
import torch
import argparse
from importlib import import_module

import numpy as np
from utils.utils import build_dataset, DatasetIterater
from utils.ngram_utils import build_ngram_dataset, NGramDatasetIterater
from scripts import init_network, train


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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

    # model name
    model_name = args.model
    if model_name == "FastText":
        build_dataset = build_ngram_dataset
        DatasetIterater = NGramDatasetIterater
    
    # module 
    module = import_module("models")
    Config = getattr(module, model_name + "Config")
    Model = getattr(module, model_name)

    # config
    config = Config(args.dataset, embedding)

    # dataset
    vocab, train_data, dev_data, test_data = build_dataset(config, args.token_level)
    config.n_vocab = len(vocab)
    train_iter = DatasetIterater(train_data, config)
    dev_iter = DatasetIterater(dev_data, config)
    test_iter = DatasetIterater(test_data, config)
    print(next(train_iter))
    
    # model
    model = Model(config).to(config.device)
        
    # network initialization
    init_network(model=model)

    # train
    train(config, model, train_iter, dev_iter, test_iter)


if __name__ == "__main__":
    main()