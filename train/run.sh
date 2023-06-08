# TextCNN
echo TextCNN
python train/run.py --model TextCNN --embedding random

# TextRNN
echo TextRNN
python train/run.py --model TextRNN --embedding random

# TextRNNAtt
echo TextRNNAtt
python train/run.py --model TextRNNAtt --embedding random

# TextRCNN
echo TextRCNN
python train/run.py --model TextRCNN --embedding random

# DPCNN
echo DPCNN
python train/run.py --model DPCNN --embedding random

# FastText
echo FastText
python train/run.py --model FastText --embedding random

# Transformer
echo Transformer
python train/run.py --model Transformer