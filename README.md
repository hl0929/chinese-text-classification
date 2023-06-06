
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

# 原始数据

[THUCNews](http://thuctc.thunlp.org/)

# 相关实现

[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)


# 单元测试

* 运行所有测试文件
```bash
nosetests -v tests/*
```

* 运行具体某一个测试文件
```bash
python -m unittest -v tests.test_text_cnn
```

# TensorBoard

```bash
 tensorboard --logdir=./
 ```

