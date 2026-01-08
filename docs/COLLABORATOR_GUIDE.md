# 合作者快速指南

欢迎查看 HyperAmy 项目的数据集、实验结果和报告！

## 📋 快速导航

### 📊 数据集
- **训练数据集**: [`data/training/monte_cristo_train_full.jsonl`](../data/training/monte_cristo_train_full.jsonl)
  - 10,000个chunks，包含情感强度、惊奇度和Mass分数
  - 文件大小: 2.61 MB
  
- **QA基准测试**: [`data/public_benchmark/monte_cristo_qa_full.json`](../data/public_benchmark/monte_cristo_qa_full.json)
  - 50个QA对，100%需要情绪敏感性
  - 文件大小: 0.05 MB

详细说明: 参见 [DATASET_STATUS.md](DATASET_STATUS.md)

### 📈 实验结果

#### 已完成的实验

1. **两方法对比 V1** (原始版本)
   - 结果文件: [`outputs/two_methods_comparison/comparison_results.json`](../outputs/two_methods_comparison/comparison_results.json)
   - 方法: HippoRAG (纯语义) vs Fusion (语义+情绪混合)
   - 规模: 3个查询（小规模测试）

2. **两方法对比 V2** (优化版本)
   - 结果文件: [`outputs/two_methods_comparison_v2/comparison_results.json`](../outputs/two_methods_comparison_v2/comparison_results.json)
   - 方法: HippoRAG (纯语义) vs Fusion (语义+情绪混合)
   - 规模: 3个查询（小规模测试）
   - 优化: 并发处理，性能提升10倍

3. **三种方法对比** (Monte Cristo数据集) 🔄
   - 结果目录: [`outputs/three_methods_comparison_monte_cristo/`](../outputs/three_methods_comparison_monte_cristo/)
   - 方法: HyperAmy (纯情绪) vs HippoRAG (纯语义) vs Fusion (混合)
   - 规模: 9,734个chunks，50个查询
   - 状态: 进行中（预计很快完成）

### 📝 重要报告

#### 必读报告

1. **[PROJECT_STATUS.md](PROJECT_STATUS.md)** ⭐ **推荐首先阅读**
   - 项目整体状态
   - 数据集和实验概览
   - 主要发现和下一步计划

2. **[EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)**
   - 完整的实验总结
   - 已完成实验详情
   - 技术优化说明
   - 实验意义和发现

3. **[DATASET_STATUS.md](DATASET_STATUS.md)**
   - 数据集完整性验证
   - 详细统计数据
   - 数据集使用说明

4. **[EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md)**
   - 实验结果详细分析
   - V1和V2批次对比
   - 方法性能分析
   - 查询类型相关性

#### 实验计划文档

1. **[THREE_METHODS_EXPERIMENT_PLAN.md](THREE_METHODS_EXPERIMENT_PLAN.md)**
   - 三种方法对比实验的详细计划
   - 实验步骤和预期结果

2. **[BATCH_EXPERIMENTS_PLAN.md](BATCH_EXPERIMENTS_PLAN.md)**
   - 分批实验计划
   - 优化对比

## 🔍 主要发现摘要

### 初步结果（基于V2实验）

| 查询类型 | HippoRAG表现 | Fusion表现 | 结论 |
|---------|-------------|-----------|------|
| **纯语义查询** | ✅ 优秀 (1.0000) | ⚠️ 一般 (0.47-0.93) | HippoRAG更适合 |
| **情绪相关查询** | ❌ 很差 (0.0090) | ✅ 良好 (0.4764) | Fusion更适合 |

**关键发现**: 
- 语义查询 → 使用HippoRAG（纯语义检索）
- 情绪查询 → 使用Fusion（语义+情绪混合检索）

**详细分析**: 参见 [EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md)

## 📊 数据格式说明

### 训练数据集格式 (JSONL)

每行一个JSON对象：
```json
{
  "input": "文本内容...",
  "emotion_intensity": 0.5,
  "surprisal": 0.8,
  "target_mass": 0.6
}
```

### QA数据集格式 (JSON)

JSON数组，每个元素：
```json
{
  "question": "查询问题...",
  "answer": "标准答案...",
  "chunk_id": "chunk_1234",
  "chunk_text": "对应的chunk文本...",
  "requires_emotional_sensitivity": true,
  "key_evidence": "...",
  "reasoning": "...",
  "mass": 0.7
}
```

### 实验结果格式 (JSON)

JSON数组，每个元素包含三种方法的结果：
```json
{
  "question": "查询问题...",
  "gold_chunk_id": "chunk_1234",
  "hipporag": {
    "available": true,
    "hit_at_1": 1,
    "top_score": 0.95,
    "docs": [...],
    "doc_scores": [...]
  },
  "fusion": {
    "available": true,
    "hit_at_1": 1,
    "top_score": 0.88,
    "docs": [...],
    "doc_scores": [...]
  },
  "hyperamy": {
    "available": true,
    "hit_at_1": 0,
    "top_score": 0.72,
    "docs": [...],
    "doc_scores": [...]
  }
}
```

## 🛠️ 如何使用数据

### 加载训练数据

```python
import json
from pathlib import Path

# 加载训练数据
chunks = []
with open('data/training/monte_cristo_train_full.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"加载了 {len(chunks)} 个chunks")
```

### 加载QA数据

```python
import json

# 加载QA数据
with open('data/public_benchmark/monte_cristo_qa_full.json', 'r') as f:
    qa_pairs = json.load(f)

print(f"加载了 {len(qa_pairs)} 个QA对")
```

### 查看实验结果

```python
import json

# 加载实验结果
with open('outputs/two_methods_comparison_v2/comparison_results.json', 'r') as f:
    results = json.load(f)

# 分析结果
for i, result in enumerate(results):
    print(f"\n查询 {i+1}: {result['question']}")
    if result['hipporag']['available']:
        print(f"  HippoRAG: Hit@1={result['hipporag']['hit_at_1']}, Score={result['hipporag']['top_score']:.4f}")
    if result['fusion']['available']:
        print(f"  Fusion: Hit@1={result['fusion']['hit_at_1']}, Score={result['fusion']['top_score']:.4f}")
```

## 📁 文件结构

```
HyperAmy/
├── data/                           # 数据集
│   ├── training/
│   │   └── monte_cristo_train_full.jsonl  # ✅ 完整训练数据集
│   └── public_benchmark/
│       └── monte_cristo_qa_full.json      # ✅ 完整QA数据集
├── outputs/                        # 实验结果
│   ├── two_methods_comparison/
│   │   └── comparison_results.json        # ✅ V1结果
│   ├── two_methods_comparison_v2/
│   │   └── comparison_results.json        # ✅ V2结果
│   └── three_methods_comparison_monte_cristo/
│       └── comparison_results.json        # 🔄 进行中
└── docs/                          # 文档和报告
    ├── PROJECT_STATUS.md          # ⭐ 项目状态（推荐首先阅读）
    ├── EXPERIMENT_SUMMARY.md      # 实验总结
    ├── DATASET_STATUS.md          # 数据集状态
    ├── EXPERIMENT_RESULTS_ANALYSIS.md  # 结果分析
    └── COLLABORATOR_GUIDE.md      # 本文件
```

## ❓ 常见问题

### Q: 数据集在哪里？
A: 数据集在 `data/` 目录下。完整训练数据集在 `data/training/monte_cristo_train_full.jsonl`，QA数据集在 `data/public_benchmark/monte_cristo_qa_full.json`。

### Q: 实验结果在哪里？
A: 实验结果在 `outputs/` 目录下。每个实验都有独立的子目录，包含 `comparison_results.json` 文件。

### Q: 如何理解实验结果？
A: 参见 [EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md) 了解详细的实验结果分析。

### Q: 三种方法对比实验完成了吗？
A: 实验正在进行中，预计很快完成。结果文件将保存在 `outputs/three_methods_comparison_monte_cristo/comparison_results.json`。

### Q: 从哪里开始看？
A: 推荐先阅读 [PROJECT_STATUS.md](PROJECT_STATUS.md) 了解整体状态，然后根据需要查看其他报告。

## 📞 需要帮助？

- 查看 [PROJECT_STATUS.md](PROJECT_STATUS.md) 了解项目整体状态
- 查看 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) 了解实验详情
- 查看 [DATASET_STATUS.md](DATASET_STATUS.md) 了解数据集详情
- 查看 [EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md) 了解结果分析

---

**最后更新**: 2026-01-08

