import sys
sys.path.append(".")

import torch
from models import TextCNN, TextCNNConfig


class Config:

    def __init__(self) -> None:
        self.embedding_pretrained = None
        self.n_vocab = 4762
        self.embedding_dim = 300
        self.pad_size = 32
        self.filter_sizes = [2, 3, 4] 
        self.num_filters = 256    
        self.dropout = 0.5 
        self.num_classes = 10


def main():
    config = Config()
    model = TextCNN(config)

    ckpt_path = "data/weights/TextCNN.ckpt"
    onnx_path = "data/weights/TextCNN.onnx"

    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randint(0, 10, (1, config.pad_size), dtype=int),
            onnx_path,
            opset_version=11,
            input_names=["input"],
            output_names=["output"]
        )
    print("------- onnx --------")
    print("To onnx done.")
    print("Save to:", onnx_path)
    print("------- onnx --------")

if __name__ == "__main__":
    main()