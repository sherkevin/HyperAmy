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

### 三种检索方法对比实验已完成！

📊 **详细分析报告**: [EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md](EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md)  
📢 **合作者通知**: [COLLABORATORS_UPDATE.md](COLLABORATORS_UPDATE.md)

**实验结果摘要**:
- ✅ **HippoRAG (纯语义)**: Recall@5 = 58.0%，在精确匹配（Recall@1-2）上表现最好
- ✅ **Fusion (语义+情绪)**: Recall@10 = 62.0% 🏆，在整体检索上表现最好
- ⚠️ **HyperAmy (纯情绪)**: 检索算法存在问题，正在修复中

## 🔍 主要发现

基于三种方法对比实验的发现：

1. **HippoRAG (纯语义)**: 在精确匹配（Recall@1-2）上表现最好，适合需要精确匹配的场景
2. **Fusion (语义+情绪)**: 在整体检索（Recall@10）上达到62%的最佳表现，证明了融合情绪特征的有效性
3. **HyperAmy (纯情绪)**: 检索算法存在问题，需要进一步修复

**使用建议**:
- 追求精确匹配（Top-1）→ 使用 **HippoRAG**
- 追求整体检索性能（Top-10）→ 使用 **Fusion**
- 情绪敏感检索 → 等待 **HyperAmy** 修复后测试

详细分析参见:
- [EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md](EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md) - 三种方法对比完整分析
- [EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md) - 两方法对比分析

---

**最后更新**: 2026-01-09
