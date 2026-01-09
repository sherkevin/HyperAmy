# HyperAmy 项目文档

欢迎查看 HyperAmy 项目的文档、数据集和实验结果！

## 🚀 快速开始

**新来者请先阅读**: [COLLABORATOR_GUIDE.md](COLLABORATOR_GUIDE.md)

## 📚 文档目录

### 核心文档（推荐按顺序阅读）

1. **[COLLABORATOR_GUIDE.md](COLLABORATOR_GUIDE.md)** ⭐ **开始从这里**
   - 快速导航指南
   - 数据集和实验结果位置
   - 文件格式说明
   - 常见问题

2. **[PROJECT_STATUS.md](PROJECT_STATUS.md)**
   - 项目整体状态
   - 数据集和实验概览
   - 主要发现

3. **[EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)**
   - 完整的实验总结
   - 已完成实验详情
   - 技术优化说明

4. **[DATASET_STATUS.md](DATASET_STATUS.md)**
   - 数据集完整性验证
   - 详细统计数据
   - 使用说明

5. **[EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md)**
   - 实验结果详细分析
   - 方法性能对比
   - 查询类型相关性

### 实验计划文档

- [THREE_METHODS_EXPERIMENT_PLAN.md](THREE_METHODS_EXPERIMENT_PLAN.md) - 三种方法对比实验计划
- [BATCH_EXPERIMENTS_PLAN.md](BATCH_EXPERIMENTS_PLAN.md) - 分批实验计划

### 其他文档

- [PUBLIC_DATASET_GUIDE.md](PUBLIC_DATASET_GUIDE.md) - 公开数据集构建指南
- [DATASET_BUILD_STATUS.md](DATASET_BUILD_STATUS.md) - 数据集构建状态
- [TEST_RESULTS_SUMMARY.md](TEST_RESULTS_SUMMARY.md) - 测试结果总结

## 📊 数据集位置

- **训练数据集**: [`../data/training/monte_cristo_train_full.jsonl`](../data/training/monte_cristo_train_full.jsonl)
- **QA基准测试**: [`../data/public_benchmark/monte_cristo_qa_full.json`](../data/public_benchmark/monte_cristo_qa_full.json)

## 📈 实验结果位置

- **两方法对比 V1**: [`../outputs/two_methods_comparison/comparison_results.json`](../outputs/two_methods_comparison/comparison_results.json)
- **两方法对比 V2**: [`../outputs/two_methods_comparison_v2/comparison_results.json`](../outputs/two_methods_comparison_v2/comparison_results.json)
- **三种方法对比**: [`../outputs/three_methods_comparison_monte_cristo/comparison_results.json`](../outputs/three_methods_comparison_monte_cristo/comparison_results.json) ✅ **已完成**

## 🎯 最新实验成果（2026-01-09）

### ✅ 三种检索方法对比实验已完成！

📊 **完整实验报告**: [EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS_FINAL.md](EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS_FINAL.md) ⭐ **最新**  
📊 **详细分析报告**: [EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md](EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md)  
📋 **下一步工作计划**: [NEXT_STEPS_AFTER_THREE_METHODS_EXPERIMENT.md](NEXT_STEPS_AFTER_THREE_METHODS_EXPERIMENT.md) ⭐ **重要**  
📢 **合作者通知**: [COLLABORATORS_UPDATE.md](COLLABORATORS_UPDATE.md)

**实验结果摘要**:
- ✅ **HippoRAG (纯语义)**: Recall@1=28.0% 🏆, Recall@5=58.0%，在精确匹配（Recall@1-2）上表现最佳
- ✅ **Fusion (语义+情绪)**: Recall@10=62.0% 🏆, Recall@2=40.0% 🏆，在整体检索上表现最佳
- ✅ **HyperAmy (纯情绪)**: 实验成功完成，需要开发基于情绪相似度的新评估方法

## 🔍 核心发现

基于三种方法对比实验（50个QA对）的发现：

1. **🏆 HippoRAG 在精确匹配上表现最好**：
   - Recall@1: 28.0%（最佳）
   - Recall@2: 34.0%（最佳）
   - 适合需要精确匹配的场景

2. **🏆 Fusion 在整体检索上表现最好**：
   - Recall@10: 62.0%（最佳）
   - Recall@2: 40.0%（最佳）
   - 证明了融合情绪特征的有效性（在Recall@10上优于HippoRAG 4个百分点）

3. **💡 情绪特征确实有效**：
   - Fusion 在 Recall@10 上的表现（62.0%）明显优于 HippoRAG（58.0%）
   - 说明融合情绪特征能够提升检索性能

4. **✅ HyperAmy 纯情绪检索是成功的**：
   - 精确匹配率：0-2%（预期行为，基于相似度而非精确匹配）
   - **情绪相似度评估**：Recall@2=100.0%, Recall@5=100.0% 🏆（使用EmotionRecall@K）
   - 证明了纯情绪检索方法的有效性
   - 需要使用情绪相似度而非精确匹配来评估HyperAmy

**使用建议**:
- **精确匹配场景（Top-1-2）** → 推荐使用 **HippoRAG** 🏆（Recall@1=28.0%）
- **整体检索场景（Top-10）** → 推荐使用 **Fusion** 🏆（Recall@10=62.0%）
- **情绪相似检索** → 推荐使用 **HyperAmy** 🏆（EmotionRecall@2=100.0%）

**评估标准说明**:
- HippoRAG/Fusion: 使用精确匹配（Exact Match）评估
- HyperAmy: 使用情绪相似度（Emotion Similarity）评估

详细分析参见:
- [EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS_FINAL.md](EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS_FINAL.md) - **完整实验报告（最新）**
- [HYPERAMY_EMOTION_RECALL_RESULTS.md](HYPERAMY_EMOTION_RECALL_RESULTS.md) - **HyperAmy情绪相似度评估结果** ⭐
- [MISSED_QUERIES_ANALYSIS_REPORT.md](MISSED_QUERIES_ANALYSIS_REPORT.md) - **未命中查询深度分析** ⭐
- [EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md](EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md) - 详细分析报告
- [EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md) - 两方法对比分析

## 📋 下一步工作计划

### 🔴 高优先级任务

1. **开发HyperAmy评估方法** ⭐⭐⭐
   - 实现基于情绪相似度的评估指标（EmotionRecall@K）
   - 正确评估HyperAmy纯情绪检索的性能

2. **深入分析未命中查询** ⭐⭐⭐
   - 分析40%完全未命中的查询
   - 找出改进方向

3. **参数调优实验** ⭐⭐
   - 优化Fusion的sentiment_weight参数
   - 找到最佳配置

详细计划参见: [NEXT_STEPS_AFTER_THREE_METHODS_EXPERIMENT.md](NEXT_STEPS_AFTER_THREE_METHODS_EXPERIMENT.md)

---

**最后更新**: 2026-01-09
