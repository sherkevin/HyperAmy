# 语义崩溃协议 (Semantic Collapse Protocol)

**实施时间**: 2026-01-19  
**状态**: ✅ **已实施**  
**问题诊断**: Vibe Search 实验中，HippoRAG 的 Recall@1 只有 12%，说明语义检索已经失效，但"最低语义保护"机制（W_sem >= 0.7）仍然强制分配 70% 权重给无效的语义结果，导致 HyperAmy 无法发挥主导作用。

---

## 🚨 问题背景

### 核心矛盾

1. **"最低语义保护"的初衷**：在 Factoid QA 任务中，情绪检索往往无效（0% Recall），需要保护语义检索的主导地位（防止 W_emo 过高导致性能下降）。

2. **Vibe Search 的场景**：语义检索已经失效（S_sem < 0.05），正确答案可能在语义检索中排到 100 名以外，如果仍然强制 W_sem = 0.7，即使 HyperAmy 找到了正确答案，也无法将其提升到 Top-1。

### 数据证据

- **实验2 (Emos GPU)**：HippoRAG Recall@1 = 12%，S_sem ≈ 0.0004-0.0008（极低）
- **当前权重**：W_emo = 0.3, W_sem = 0.7（被最低语义保护锁定）
- **结果**：HyperAmy Recall@1 = 10%，几乎等于基线

---

## ✅ 解决方案：语义崩溃协议

### 核心逻辑

```python
SEMANTIC_COLLAPSE_THRESHOLD = 0.05  # 语义崩溃阈值

if S_sem < SEMANTIC_COLLAPSE_THRESHOLD:
    # 语义崩溃，解除安全锁！
    # 权重完全由情绪强度决定，最高可达 0.95（仍保留5%语义作为兜底）
    w_emo = 0.5 + (I_q * 0.45)  # I_q=1.0时，w_emo=0.95
    w_sem = 1.0 - w_emo  # 剩余权重给语义（最低5%兜底）
else:
    # 正常情况：保持原有的保护逻辑（W_sem >= 0.7）
    # ...
```

### 权重计算规则

| S_sem 范围 | I_q | W_emo | W_sem | 说明 |
|------------|-----|-------|-------|------|
| **< 0.05** (崩溃) | 0.5 | 0.73 | 0.27 | 语义崩溃，情绪主导 |
| **< 0.05** (崩溃) | 0.8 | 0.86 | 0.14 | 语义崩溃 + 高情绪强度 |
| **< 0.05** (崩溃) | 1.0 | 0.95 | 0.05 | 语义崩溃 + 极高情绪强度 |
| **>= 0.05** (正常) | 任意 | ≤ 0.3 | ≥ 0.7 | 保持最低语义保护 |

### 预期效果

在 Vibe Search 场景下（S_sem ≈ 0.0004, I_q ≈ 0.8-0.9）：

- **修正前**：W_emo = 0.3, W_sem = 0.7 → HyperAmy Recall@1 = 10%
- **修正后**：W_emo = 0.86, W_sem = 0.14 → HyperAmy Recall@1 **> 70%**（预期）

---

## 📊 实施细节

### 代码位置

- **文件**：`poincare/retrieval.py`
- **方法**：`HyperAmyRetrieval.search_hybrid()`
- **行数**：约 1037-1070 行

### 关键修改

1. **添加语义崩溃检测**：`if S_sem < SEMANTIC_COLLAPSE_THRESHOLD:`
2. **解除安全锁**：在语义崩溃时，允许 W_emo 达到 0.95
3. **保留兜底机制**：即使在语义崩溃时，仍保留至少 5% 的语义权重（防止完全失控）
4. **日志记录**：添加警告日志，便于监控语义崩溃事件

### 日志示例

```
⚠️ Semantic Collapse Detected (S_sem=0.0004 < 0.05)! Releasing Safety Lock (I_q=0.88).
Dynamic Weighting: Iq=0.8800, S_sem=0.0004 -> Base=0.9972, Supp=1.0000 -> Final W_emo=0.8960, W_sem=0.1040
```

---

## 🧪 验证方案

### 测试场景

1. **Vibe Search 数据集**：语义检索失效（S_sem < 0.05）
   - **预期**：W_emo 应该达到 0.8 以上
   - **预期**：HyperAmy Recall@1 应该 > 70%

2. **Factoid QA 数据集**：语义检索有效（S_sem > 0.05）
   - **预期**：保持最低语义保护（W_sem >= 0.7）
   - **预期**：HyperAmy Recall@1 应该 ≥ 基线（防止性能下降）

### 评估指标

- **Recall@1**：Top-1 命中率（主要指标）
- **W_emo 分布**：统计所有查询的平均 W_emo
- **语义崩溃事件数**：统计触发语义崩溃协议的查询数量

---

## 📈 预期结果

| 方法 | 修正前 | 修正后 (预期) |
|------|--------|--------------|
| **HippoRAG** | 12% | 12% (保持不变) |
| **HyperAmy** | 10% | **> 70%** |
| **Hybrid** | 10% | **> 80%** |

---

## ⚠️ 注意事项

1. **阈值选择**：`SEMANTIC_COLLAPSE_THRESHOLD = 0.05` 是基于实验观察选择的，可能需要根据实际情况调整。

2. **兜底机制**：即使在语义崩溃时，仍保留至少 5% 的语义权重，这是为了：
   - 防止完全失控（纯情绪检索可能在某些边缘情况下失效）
   - 保持系统的鲁棒性

3. **向后兼容**：在正常场景下（S_sem >= 0.05），系统行为与之前完全一致，不会影响 Factoid QA 任务的性能。

---

## 🔗 相关文档

- [Dynamic Weighting Hybrid Search](./DYNAMIC_WEIGHTING_HYBRID_SEARCH.md)
- [Vibe Search Experiment Monitoring](./VIBE_SEARCH_EXPERIMENT_MONITORING.md)
- [Emotional Orthogonality Hypothesis](./EMOTIONAL_ORTHOGONALITY_HYPOTHESIS.md)
