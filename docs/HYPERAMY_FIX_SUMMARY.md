# HyperAmy 检索问题修复总结

**修复日期**: 2026-01-09  
**问题状态**: ✅ 已修复

## 问题描述

在三种检索方法对比实验中，HyperAmy（纯情绪检索）的检索结果中**没有包含任何一个 gold_chunk_id**，导致 Recall@5 = 0%。

虽然所有 gold_chunk_id 都在索引中（100%覆盖率），但检索算法未能正确匹配。

## 根本原因

经过诊断，发现问题出在：

1. **使用了已移除的 `Point` 类**：
   - 实验脚本中使用了 `from poincare.types import Point`
   - 但 `Point` 类已在代码重构中移除，`poincare` 模块现在直接使用 `particle.ParticleEntity`

2. **调用了不存在的方法**：
   - 脚本中使用了 `storage.upsert_point(point)`
   - 但 `HyperAmyStorage` 只有 `upsert_entity(entity)` 方法，没有 `upsert_point()` 方法

3. **接口类型不匹配**：
   - `HyperAmyRetrieval.search()` 需要 `ParticleEntity` 对象
   - 但脚本中传入的是 `Point` 对象
   - `ParticleEntity` 和 `Point` 的属性结构完全不同

## 修复方案

### 1. 替换数据类型

**之前**:
```python
from poincare.types import Point

point = Point(
    id=chunk_id,
    emotion_vector=emotion_vector,  # torch.Tensor
    v=0.0,
    T=1.0,
    born=0.0
)
storage.upsert_point(point)
```

**修复后**:
```python
from particle.particle import ParticleEntity

# 确保emotion_vector是numpy array并归一化
if isinstance(emotion_vector, torch.Tensor):
    emotion_vector = emotion_vector.cpu().numpy()
elif not isinstance(emotion_vector, np.ndarray):
    emotion_vector = np.array(emotion_vector, dtype=np.float32)

# 计算weight（原始情绪向量的模长）
weight = float(np.linalg.norm(emotion_vector))

# 归一化情绪向量（ParticleEntity期望归一化后的向量）
if weight > 1e-9:
    normalized_vector = emotion_vector / weight
else:
    normalized_vector = emotion_vector.copy()
    weight = 0.0

entity = ParticleEntity(
    entity_id=chunk_id,
    entity=f"chunk_{chunk_idx}",
    text_id=chunk_id,
    emotion_vector=normalized_vector,  # 归一化后的numpy array
    weight=weight,  # 原始模长
    speed=0.0,
    temperature=1.0,
    born=0.0
)
storage.upsert_entity(entity)
```

### 2. 修复检索调用

**之前**:
```python
query_point = Point(
    id=query_id,
    emotion_vector=query_emotion,  # torch.Tensor
    v=0.0,
    T=1.0,
    born=0.0
)
search_results = hyperamy_retrieval.search(query_point, top_k=5)
```

**修复后**:
```python
# 归一化查询情绪向量
if isinstance(query_emotion, torch.Tensor):
    query_emotion = query_emotion.cpu().numpy()
elif not isinstance(query_emotion, np.ndarray):
    query_emotion = np.array(query_emotion, dtype=np.float32)

weight = float(np.linalg.norm(query_emotion))
if weight > 1e-9:
    normalized_vector = query_emotion / weight
else:
    normalized_vector = query_emotion.copy()
    weight = 0.0

query_entity = ParticleEntity(
    entity_id=query_id,
    entity=query[:50],
    text_id=f"query_{i}",
    emotion_vector=normalized_vector,
    weight=weight,
    speed=0.0,
    temperature=1.0,
    born=0.0
)
search_results = hyperamy_retrieval.search(query_entity, top_k=5)
```

## 修复的文件

1. **`test/test_three_methods_comparison_monte_cristo.py`** - 主实验脚本
   - 修复了存储逻辑
   - 修复了检索逻辑

2. **`test/test_hyperamy_parallel.py`** - 并行索引脚本
   - 修复了存储逻辑（与主脚本保持一致）

## 关键差异

### Point vs ParticleEntity

| 属性 | Point (已移除) | ParticleEntity |
|------|---------------|----------------|
| ID | `id: str` | `entity_id: str` |
| 实体名称 | 无 | `entity: str` |
| 文本ID | 无 | `text_id: str` |
| 情绪向量 | `emotion_vector: torch.Tensor` | `emotion_vector: np.ndarray` (归一化) |
| 权重 | 无 | `weight: float` (原始模长) |
| 速度 | `v: float` | `speed: float` |
| 温度 | `T: float` | `temperature: float` |
| 创建时间 | `born: float` | `born: float` |

### 重要注意事项

1. **向量归一化**：
   - `ParticleEntity` 期望 `emotion_vector` 是**归一化后的 numpy array**
   - 需要计算 `weight`（原始向量的模长）并单独存储

2. **方法调用**：
   - 存储：`storage.upsert_entity(entity)` ✅
   - ~~`storage.upsert_point(point)` ❌~~ (不存在)

3. **检索参数**：
   - `retrieval.search(query_entity, ...)` ✅
   - `query_entity` 必须是 `ParticleEntity` 对象

## 验证建议

修复后，建议：

1. **重新运行小规模测试**（如 10 个查询）验证修复是否有效
2. **检查检索结果**：
   - 确认检索结果中包含 gold_chunk_id
   - 计算 Recall@K 指标，应该 > 0%
3. **如果验证成功，重新运行完整实验**（50 个查询）

## 提交记录

- **Commit 1**: `2ab39a9` - fix: 修复HyperAmy检索问题 - 使用ParticleEntity替代已移除的Point类
- **Commit 2**: `855d23d` - fix: 修复HyperAmy并行索引脚本 - 使用ParticleEntity替代Point类

## 相关文档

- [三种方法对比实验分析报告](./EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md)
- [合作者通知](./COLLABORATORS_UPDATE.md)

---

**修复完成时间**: 2026-01-09  
**下一步**: 验证修复效果，重新运行实验

