# 实体粒度情绪数据集v2 - 数据集已就绪

## 📋 概述

实体粒度情绪数据集v2已成功生成并完成质量验证。该数据集使用改进的生成脚本（v2），解决了之前版本中的关键问题，并添加了QA对数据支持。

## 📊 数据集统计信息

- **文件位置**: `data/training/entity_granularity/entity_granularity_v2_full.jsonl`
- **样本数量**: 1,464个样本
- **总实体数**: 约1,800+个实体
- **文件大小**: 约1.5-2MB
- **数据来源**: 
  - Monte Cristo训练文本（10,000个文本）
  - Monte Cristo QA对（50个QA对，生成100个文本）

## ✅ 质量验证结果

### 1. 格式正确性 ✅
- 所有样本格式正确，符合规范
- 所有字段（text, targets, span_text, char_start, char_end, soft_label, intensity）完整
- 字符位置匹配正确

### 2. soft_label格式 ✅
- **维度**: 28维（正确）
- **范围**: 每个维度在[0, 1]范围内
- **归一化**: 代码不强制归一化，保持模长可变（符合需求）
- **说明**: LLM返回的soft_label通常是归一化的（sum≈1.0），这是正常的，代码不会再次归一化

### 3. intensity计算 ✅
- **方法**: 使用L2-norm（更合理的强度度量）
- **验证**: 与soft_label的L2-norm匹配
- **改进**: 相比v1版本的max-norm，L2-norm能更好地反映情绪强度

### 4. 章节标题过滤 ✅
- 章节标题（如"Marseilles—The Arrival\nChapter 2"）已被有效过滤
- 包含"Chapter"的样本比例 < 1%

### 5. 文本质量 ✅
- 平均文本长度合理（>100字符）
- 短文本（<50字符）比例低
- 文本内容质量良好

## 🔧 关键改进点

### 相比v1版本的改进：

1. **soft_label不归一化**
   - v1: 强制归一化（sum=1.0），导致所有向量模长固定
   - v2: 不归一化，每个维度在[0,1]，模长可变，能反映真实的情绪强度变化

2. **intensity使用L2-norm**
   - v1: 使用max-norm（最大值）
   - v2: 使用L2-norm（更合理的强度度量）

3. **章节标题过滤**
   - v1: 包含大量章节标题，导致实体提取质量差
   - v2: 智能过滤章节标题，提升数据质量

4. **QA对数据支持**
   - v1: 仅使用训练文本
   - v2: 支持QA对数据，确保问题和答案的实体情绪向量相近，提升模型在QA检索任务上的性能

## 📝 数据格式

```json
{
  "text": "The cat ate the cat food.",
  "targets": [
    {
      "span_text": "cat",
      "char_start": 4,
      "char_end": 7,
      "soft_label": [0.01, 0.02, 0.85, 0.05, ...],  // 28-dim vector (each in [0,1], NOT normalized)
      "intensity": 0.92  // L2-norm of soft_label
    }
  ]
}
```

### 字段说明：
- `text`: 原始文本
- `targets`: 实体列表
  - `span_text`: 实体文本
  - `char_start`, `char_end`: 字符位置（精确匹配）
  - `soft_label`: 28维情绪向量（每个维度在[0,1]，不归一化）
  - `intensity`: 情绪强度（L2-norm）

## 🎯 使用建议

### 训练emotion embedding model

1. **数据加载**:
   ```python
   import json
   from pathlib import Path
   
   dataset_file = Path("data/training/entity_granularity/entity_granularity_v2_full.jsonl")
   samples = []
   with open(dataset_file, 'r', encoding='utf-8') as f:
       for line in f:
           if line.strip():
               samples.append(json.loads(line))
   ```

2. **训练目标**:
   - 输入：实体文本（span_text）
   - 输出：28维情绪向量（soft_label）
   - 损失函数：考虑intensity（L2-norm），保持模长可变

3. **关键点**:
   - soft_label不归一化，模长可以变化
   - intensity使用L2-norm，反映真实强度
   - QA对数据有助于模型在问答场景中的表现

## 📁 相关文件

- **数据集文件**: `data/training/entity_granularity/entity_granularity_v2_full.jsonl`
- **生成脚本**: `scripts/generate_entity_granularity_dataset_v2.py`
- **验证脚本**: `scripts/validate_entity_granularity_dataset_v2.py`
- **技术文档**: `docs/ENTITY_GRANULARITY_DATASET_V2_IMPROVEMENTS.md`
- **更新说明**: `docs/COLLABORATOR_UPDATE_ENTITY_GRANULARITY_V2.md`

## ✅ 验证命令

如需重新验证数据集质量：

```bash
python scripts/validate_entity_granularity_dataset_v2.py \
  --dataset data/training/entity_granularity/entity_granularity_v2_full.jsonl
```

## 🚀 下一步

数据集已就绪，可以开始训练emotion embedding model。建议：

1. 使用该数据集训练emotion embedding model
2. 验证模型在情绪向量提取任务上的性能
3. 测试模型在QA检索任务上的表现（受益于QA对数据）

---

**生成日期**: 2026-01-11  
**版本**: v2  
**状态**: ✅ 已验证，可用于训练  
**联系人**: 如有问题，请随时联系
