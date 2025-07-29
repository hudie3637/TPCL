import random
import numpy as np
import torch
import os
class Config:
    # 数据路径
    data_root = ""
    label_path = ""
    tumor_analysis_path = ""  
    # 模型参数
    img_size = 224
    batch_size = 8
    lr = 1e-5
    epochs = 50
    num_classes = 2
    weight_decay=1e-5
    n_splits = 5
    # 文本参数
    max_prompt_length = 512
    base_clinical_features = ['

    ]
    special_clinical_features = [
           ]
    # 新增：MedCLIPVisionModelViT 的 checkpoint 路径
    medclip_vision_checkpoint = ''
    
    # 随机种子设置
    seed = 42
    
    @classmethod
    def set_seed(cls, seed=None):
        """设置随机种子以确保实验可重复性"""
        if seed is None:
            seed = cls.seed
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"随机种子已设置为: {seed}")
    
    def __init__(self):
        """初始化配置时自动设置随机种子"""
        self.set_seed()
