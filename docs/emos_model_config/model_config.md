# emos情绪嵌入模型参数配置

## 模型信息

- **模型名称**: emos (Probabilistic G-BERT V4)
- **基础模型**: roberta-base
- **嵌入维度**: 64
- **情绪类别数**: 28
- **最大序列长度**: 128

## 训练配置

- **训练样本数**: 1317 (1800 实体)
- **验证样本数**: 147 (192 实体)
- **Batch Size**: 16
- **Effective Batch Size**: 64
- **Epochs**: 10
- **学习率 (Backbone)**: 2e-05
- **学习率 (Heads)**: 0.0001
- **Weight Decay**: 0.01
- **Warmup Ratio**: 0.1
- **Early Stopping Patience**: 3

## 训练结果

- **最佳验证Loss**: 11.21 (Epoch 8)
- **最终训练Loss**: 41.88
- **最终验证Loss**: 11.33
- **平均Kappa**: 3.69
- **Top-1准确率**: 82.81%
- **Top-3准确率**: 83.85%
- **Top-5准确率**: 83.85%

## 模型文件

- **Checkpoint路径**: /public/jiangh/emos/checkpoints/best_model.pt
- **模型大小**: 477 MB
- **训练日志**: /public/jiangh/emos/logs/train_full_*.log

