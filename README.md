
# 项目介绍

一行命令完成中文文本分类模型训练、**部署**。

# 项目环境

* 虚拟环境
```bash
python -m venv venv

source venv/bin/activate
```

* 安装依赖
```bash
pip install -r requirements.txt
```


# 训练脚本

```bash
sh ./train/run.sh 
```

# 单元测试

* 运行所有测试文件
```bash
nosetests -v tests/*
```

* 运行具体某一个测试文件
```bash
python -m unittest -v tests.test_text_cnn
```


# 原始数据

[THUCNews](http://thuctc.thunlp.org/)


# 模型实现

[text_classification_AI100](https://github.com/lc222/text_classification_AI100)

[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)

[ChineseTextClassifier](https://github.com/ami66/ChineseTextClassifier)

[Keras-TextClassification](https://github.com/yongzhuo/Keras-TextClassification)

[bert-Chinese-classification-task](https://github.com/NLPScott/bert-Chinese-classification-task)

[text-classification](https://github.com/wavewangyue/text-classification)

[text_classification](https://github.com/brightmart/text_classification)


# TensorBoard 可视化

执行如下命令：

```bash
 tensorboard --logdir=./logdir  # ./logdir 日志文件所在目录
 ```

