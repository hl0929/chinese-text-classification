# 虚拟环境
python -m venv venv
source venv/bin/activate

# 依赖安装
pip install -r requirements.txt

# 模型训练
sh ./train/run.sh 

# 服务部署
sh ./service/deploy.sh