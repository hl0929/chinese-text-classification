#!/usr/bin/bash

# 虚拟环境
python3 -m venv venv
source venv/bin/activate

# 依赖安装
pip install -r requirements.txt

# 模型训练
bash ./train/run.sh 

# 模型转换
bash ./inference/transform.sh

# 服务部署
bash ./service/deploy.sh