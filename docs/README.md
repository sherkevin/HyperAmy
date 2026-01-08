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
- **三种方法对比**: [`../outputs/three_methods_comparison_monte_cristo/`](../outputs/three_methods_comparison_monte_cristo/) (进行中)

## 🔍 主要发现

基于已完成实验的初步发现：

- **语义查询** → 使用 HippoRAG（纯语义检索）
- **情绪查询** → 使用 Fusion（语义+情绪混合检索）

详细分析参见 [EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md)

---

**最后更新**: 2026-01-08
