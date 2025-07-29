import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from base_config import Config
from trainer import CTTrainer

if __name__ == "__main__":
    config = Config()          # 加载配置（会自动设置随机种子）
    trainer = CTTrainer(config) # 初始化训练器
    trainer.run()              # 启动训练
