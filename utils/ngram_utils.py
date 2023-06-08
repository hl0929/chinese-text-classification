import sys
sys.path.append(".")

import os
import torch
import pickle
from typing import Dict, List, Tuple
from tqdm import tqdm
from utils.utils import DatasetIterater, build_vocab, tokenizer, read_txt, PAD, UNK


def bigram_hash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def trigram_hash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets


def load_dataset(path: str, token_level: str, vocab: Dict[str, int], n_gram_vocab_size: int, pad_size: int=32) -> List[Tuple[str, int]]:
    data = read_txt(path)
    contents = []
    print("load dataset")
    for line in tqdm(data):
        content, label = line.split("\t")
        tokens = tokenizer(content, token_level)
        seq_len = len(tokens)
        if pad_size:
            if pad_size > seq_len:
                tokens.extend([PAD] * (pad_size - seq_len))
            else:
                tokens = tokens[:pad_size]
        token_list = [vocab.get(token, vocab.get(UNK)) for token in tokens]
        # ngram
        buckets = n_gram_vocab_size
        bigram = [bigram_hash(token_list, i, buckets) for i in range(pad_size)]
        trigram = [trigram_hash(token_list, i, buckets) for i in range(pad_size)]
        contents.append((token_list, bigram, trigram, int(label)))
    return contents


def build_ngram_dataset(config, token_level: str) -> Tuple[Dict, List, List, List]:
    if os.path.exists(config.vocab_path):
        vocab = pickle.load(open(config.vocab_path, "rb"))
    else:
        data = [i[0] for i in read_txt(config.train_path)]
        vocab = build_vocab(data, token_level)
        pickle.dump(vocab, open(config.vocab_path, "wb"))
        print("build vocab: ", len(vocab))
        print("save vocab to: ", config.vocab_path)

    train = load_dataset(config.train_path, token_level, vocab, config.ngram_vocab, config.pad_size)
    dev = load_dataset(config.dev_path, token_level, vocab, config.ngram_vocab, config.pad_size)
    test = load_dataset(config.test_path, token_level, vocab, config.ngram_vocab, config.pad_size)
    return vocab, train, dev, test



class NGramDatasetIterater(DatasetIterater):

    def __init__(self, batches, config) -> None:
        super().__init__(batches, config)

    def _to_tensor(self, batch):
        x = torch.LongTensor([i[0] for i in batch]).to(self.device)
        bigram = torch.LongTensor([i[1] for i in batch]).to(self.device)
        trigram = torch.LongTensor([i[2] for i in batch]).to(self.device)
        y = torch.LongTensor([i[3] for i in batch]).to(self.device)
        return (x, bigram, trigram), y
