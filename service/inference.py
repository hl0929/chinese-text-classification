import torch
import pickle
import onnxruntime
import numpy as np


class ModelSerivce:
    UNK, PAD = '<UNK>', '<PAD>'

    def __init__(self) -> None:
        self.vocab = self._get_vocab()
        self.ort_session = self._get_session()
        self.labels = self._get_labels()
        
    def _get_vocab(self):
        vocab_path = "data/datasets/THUCNews/vocab/vocab.pkl"
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        return vocab
    
    def _get_session(self):
        model_path = "data/weights/TextCNN.onnx"
        session = onnxruntime.InferenceSession(model_path)
        return session
    
    def _get_labels(self):
        label_path = "data/datasets/THUCNews/class.txt"
        with open(label_path, encoding="utf-8") as f:
            class_list = [x.strip() for x in f.readlines()]
        return class_list
    
    def _tokenizer(self, text, pad_size=32):
        tokens = [i for i in text]
        if pad_size:
            if len(tokens) < pad_size:
                tokens.extend([ModelSerivce.PAD] * (pad_size - len(tokens)))
            else:
                tokens = tokens[:pad_size]
        word_list = []
        for word in tokens:
            word_list.append(self.vocab.get(word, self.vocab.get(ModelSerivce.UNK)))
        return torch.IntTensor([word_list])

    def predict(self, query):
        tokens = self._tokenizer(query)
        tokens = tokens.numpy().astype(np.int64)
        ort_inputs = {'input': tokens}
        ort_output = self.ort_session.run(['output'], ort_inputs)[0]
        index = np.argmax(ort_output)
        return self.labels[index]
    

if __name__ == "__main__":
    service = ModelSerivce()
    query = "姚明是谁"
    predict = service.predict(query)
    print(predict)