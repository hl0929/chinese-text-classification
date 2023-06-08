
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


# 相关实现

[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)



# TensorBoard 可视化

执行如下命令：

```bash
 tensorboard --logdir=./logdir  # ./logdir 日志文件所在目录
 ```

