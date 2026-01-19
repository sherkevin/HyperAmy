# 重要更新：语义崩溃协议与Experiment 3实现

**日期**: 2026-01-19  
**作者**: HyperAmy Team  
**状态**: ✅ 代码已实现，待完整测试

---

## 🎯 核心更新概述

本次更新实现了**语义崩溃协议 (Semantic Collapse Protocol)**，这是一个关键的机制，用于在语义检索完全失效时自动提升情绪检索的权重。结合**Experiment 3**的实现，我们验证了LLM API情绪提取与语义崩溃协议协同工作的效果。

---

## 🔑 关键创新：语义崩溃协议

### 问题背景

在之前的Vibe Search实验中，我们发现：
- HippoRAG（纯语义检索）在Vibe Search数据集上Recall@1只有12%
- 这说明语义检索已经"失效"（语义真空场景）
- 但"最低语义保护"机制（W_sem >= 0.7）仍然强制分配70%权重给无效的语义结果
- 导致HyperAmy（情绪检索）即使找到了正确答案，也无法将其提升到Top-1

### 解决方案

在 `poincare/retrieval.py` 的 `search_hybrid` 方法中，我们实现了**语义崩溃协议**：

```python
SEMANTIC_COLLAPSE_THRESHOLD = 0.05  # 语义崩溃阈值

if S_sem < SEMANTIC_COLLAPSE_THRESHOLD:
    # 语义崩溃，解除安全锁！
    # 权重完全由情绪强度决定，最高可达 0.95
    w_emo = 0.5 + (I_q * 0.45)  # I_q=1.0时，w_emo=0.95
    w_sem = 1.0 - w_emo
else:
    # 正常情况：保持原有的保护逻辑（W_sem >= 0.7）
    # ...
```

### 预期效果

- **I_q = 0.8**: W_emo = 0.86, W_sem = 0.14
- **I_q = 0.9**: W_emo = 0.905, W_sem = 0.095
- **I_q = 1.0**: W_emo = 0.95, W_sem = 0.05

这使得在语义失效场景下，情绪检索能够真正主导检索结果。

---

## 📊 Experiment 3: Final Fusion

### 实验设计

我们创建了 `test/test_vibe_search_experiment_3_final.py`，结合了：

1. **语义崩溃协议**（已验证机制有效）
2. **LLM API情绪提取**（I_q值高，约0.8-0.9）
3. **自动缓存复用**（节省API费用）

### 实验结果（初步）

- **语义崩溃协议触发**: 50/50 查询（100%）
- **平均I_q**: 0.7081（LLM API提取，高于Emos模型的0.376）
- **平均S_sem**: 0.0010（确认语义失效）
- **平均W_emo**: 0.8187（从0.3提升到0.82，提升173%）

### 当前状态

代码已实现并通过初步验证，但Recall@1结果需要进一步调试（映射问题已修复，待重新运行）。

---

## 📁 新增/修改的关键文件

### 核心实现
- **`poincare/retrieval.py`**: 
  - 实现了语义崩溃协议（`SEMANTIC_COLLAPSE_THRESHOLD = 0.05`）
  - 添加了警告日志和详细决策记录

### 实验脚本
- **`test/test_vibe_search_experiment_3_final.py`**: 
  - Experiment 3完整实现
  - 使用LLM API提取情绪向量
  - 自动复用缓存
  - 详细记录每条查询的决策过程

- **`scripts/run_experiment_3_final.sh`**: 
  - Experiment 3启动脚本

### 文档
- **`docs/SEMANTIC_COLLAPSE_PROTOCOL.md`**: 
  - 语义崩溃协议的完整文档
  - 包含设计思路、实现细节和预期效果

### 工具脚本
- **`scripts/monitor_semantic_collapse.sh`**: 
  - 实时监控语义崩溃协议触发
- **`scripts/monitor_both_experiments_live.sh`**: 
  - 监控多个实验的实时状态
- **`scripts/compare_both_experiments.sh`**: 
  - 对比不同实验的结果

---

## 🔬 实验对比总结

### Experiment 1 (LLM API版本)
- HippoRAG Recall@1: 32%
- HyperAmy Recall@1: 32%
- 说明：使用LLM API提取情绪，但可能使用了不同的数据集或评估方式

### Experiment 2 (Emos GPU版本)
- HippoRAG Recall@1: 12%
- HyperAmy Recall@1: 8%
- I_q平均值: 0.376（Emos模型提取，较低）
- 说明：验证了语义崩溃协议机制，但I_q值过低限制了性能提升

### Experiment 3 (LLM API + 语义崩溃协议)
- HippoRAG Recall@1: 待重新运行
- HyperAmy Recall@1: 待重新运行
- I_q平均值: 0.7081（LLM API提取，较高）
- W_emo平均值: 0.8187（已成功提升）
- 预期：Recall@1 > 75%

---

## 💡 关键洞察

1. **I_q值的重要性**：
   - LLM API提取的I_q值（0.7-0.9）远高于Emos模型（0.3-0.4）
   - 高I_q值是情绪检索成功的关键

2. **语义崩溃协议的必要性**：
   - 在Vibe Search场景下，语义检索完全失效（S_sem < 0.05）
   - 必须解除"最低语义保护"，允许情绪检索主导

3. **动态自适应的价值**：
   - 系统能够根据查询特征自动调整权重
   - 在Factoid QA中保护语义（W_sem >= 0.7）
   - 在Vibe Search中激活情绪（W_emo > 0.8）

---

## 🚀 下一步工作

1. **重新运行Experiment 3**：
   - 修复映射问题后，验证完整结果
   - 预期Recall@1 > 75%

2. **结果分析**：
   - 对比三种实验设置
   - 分析I_q值与Recall的关系
   - 验证语义崩溃协议的效果

3. **论文撰写**：
   - 记录"双重分离"实验结果
   - Factoid QA：语义主导（验证保护机制）
   - Vibe Search：情绪主导（验证激活机制）

---

## 📝 代码审查建议

### 重点关注的代码区域

1. **`poincare/retrieval.py` (lines 1037-1075)**:
   - 语义崩溃协议的核心实现
   - 权重计算逻辑
   - 日志记录

2. **`test/test_vibe_search_experiment_3_final.py`**:
   - 情绪提取逻辑（LLM API）
   - 决策信息提取（从metadata）
   - 结果评估和统计

3. **`docs/SEMANTIC_COLLAPSE_PROTOCOL.md`**:
   - 完整的设计文档
   - 预期效果和验证方案

---

## ❓ 讨论问题

1. **阈值选择**：`SEMANTIC_COLLAPSE_THRESHOLD = 0.05` 是否合理？是否需要根据数据集调整？

2. **权重公式**：`W_emo = 0.5 + (I_q * 0.45)` 中的系数（0.5和0.45）是否需要优化？

3. **兜底机制**：即使在语义崩溃时仍保留5%的语义权重，这是否足够？

4. **缓存策略**：当前使用`.cache/emotion_vectors`，是否需要考虑缓存失效和版本管理？

---

## 📞 联系

如有任何问题或建议，请随时讨论。期待大家的反馈！

---

**注意**：本次更新包含大量实验代码和工具脚本，建议在测试环境中先验证再合并到主分支。
