import os
import time
import pickle
from typing import List, Dict, Tuple
from collections import Counter
from datetime import timedelta

import jieba
import torch
from tqdm import tqdm


UNK, PAD = "<UNK>", "<PAD>"


def tokenizer(text: str, token_level: str) -> List[str]:
    if token_level == "char":
        result = [c for c in text]
    elif token_level == "word":
        result = jieba.lcut(text)
    else:
        raise Exception("token_level is char or word")
    return result


def build_vocab(texts: List[str], token_level, max_size: int=10000, min_freq: int=1) -> Dict[str, int]:
    vocab_list = []
    print("build vocab")
    for line in tqdm(texts):
        vocab_list.extend(tokenizer(line, token_level))
    sorted_vocab_list = [i[0] for i in Counter(vocab_list).most_common() if i[1] >= min_freq][:max_size]
    vocab_dic = {word: idx for idx, word in enumerate(sorted_vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def read_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = [i.strip() for i in f.readlines() if i.strip()]
    return data


def load_dataset(path: str, token_level: str, vocab: Dict[str, int], pad_size: int=32) -> List[Tuple[str, int]]:
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
        contents.append((token_list, int(label)))
    return contents


def build_dataset(config, token_level: str) -> Tuple[Dict, List, List, List]:
    if os.path.exists(config.vocab_path):
        vocab = pickle.load(open(config.vocab_path, "rb"))
    else:
        data = [i[0] for i in read_txt(config.train_path)]
        vocab = build_vocab(data, token_level)
        pickle.dump(vocab, open(config.vocab_path, "wb"))
        print("build vocab: ", len(vocab))
        print("save vocab to: ", config.vocab_path)

    train = load_dataset(config.train_path, token_level, vocab, config.pad_size)
    dev = load_dataset(config.dev_path, token_level, vocab, config.pad_size)
    test = load_dataset(config.test_path, token_level, vocab, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater:

    def __init__(self, batches, config) -> None:
        self.batches = batches
        self.batch_size = config.batch_size
        self.n_batch = len(batches) // config.batch_size
        self.residue = False
        if len(batches) % self.n_batch:
            self.residue = True
        self.index = 0
        self.device = config.device

    def _to_tensor(self, batch):
        x = torch.LongTensor([i[0] for i in batch]).to(self.device)
        y = torch.LongTensor([i[1] for i in batch]).to(self.device)
        return x, y
    
    def __next__(self):
        if self.residue and self.index == self.n_batch:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batch:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.n_batch + 1
        else:
            return self.n_batch


def get_time_diff(start_time):
    end_time = time.perf_counter()
    return timedelta(seconds=round(end_time - start_time))


if __name__ == "__main__":
    query = "这是一个测试"
    result = tokenizer(query, "char")
    print("char-level: ", result)
    result = tokenizer(query, "word")
    print("word-level: ", result)

    texts = ["这是", "测试", "测试"]
    vocabs = build_vocab(texts, "char")
    print(vocabs)