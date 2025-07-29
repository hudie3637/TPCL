# TPCL: Tri-modal Phase-aware Contrastive Learning for Multiphase CT, Clinical Data, and Medical Text Integration

本项目实现了论文《TPCL: A Tri-modal Phase-aware Contrastive Learning Framework for Multiphase CT, Clinical Data, and Medical Text Integration》中提出的三模态期相感知对比学习框架（TPCL），用于多期相CT、结构化临床数据和医学文本的深度融合与肝细胞癌（HCC）辅助诊断。

## 目录结构

```
.
├── base_config.py         # 配置文件，包含数据集路径、模型参数、特征定义等
├── clinical_vit.py        # 主模型实现（ClinicalViT），三模态融合及期相感知机制
├── data_loader.py         # 数据加载与预处理，支持多期相图像、临床数据、文本
├── main.py                # 入口脚本，可自定义训练/推理流程
├── metrics.py             # 评估指标封装
├── mode.py                # 期相感知注意力融合网络（PAAF）实现
├── trainer.py             # 训练与验证流程、损失函数组合、评价输出
```

## 环境与依赖

- Python ≥ 3.7
- PyTorch ≥ 1.10
- torchvision
- pandas
- scikit-learn
- tqdm
- PIL
- 以及medclip医学预训练权重 （https://storage.googleapis.com/pytrial/medclip-pretrained.zip）


## 快速开始

1. **准备数据**  
   - 多期相CT图片，按结构`data_root/放射号+时间点/期相/*.jpg`组织（期相如SP, AP, VP, DP）。
   - 临床数据（Excel），以及肿瘤分析文本（JSON）。
   - 修改`base_config.py`中的路径配置。

2. **训练模型**

```bash
python main.py
```

3. **评估与结果输出**  
   训练/验证结束后，预测结果与真实标签对比将保存在`predictions_vs_labels.csv`。






