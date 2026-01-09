# HyperAmy修复验证计划

**日期**: 2026-01-09  
**状态**: 📋 准备执行

## 验证步骤

### 步骤1: 小规模测试（10个查询）

**目的**: 快速验证修复是否有效

**执行方式**:
```bash
# 在实验服务器上运行
ssh hyperamy-server
cd /public/jiangh/HyperAmy
bash scripts/run_hyperamy_validation.sh
```

**预期结果**:
- HyperAmy能够正确检索到gold_chunk_id
- Recall@5 > 0%
- 每个查询都能返回检索结果

**验证指标**:
- Recall@1, Recall@2, Recall@5, Recall@10
- 命中数量（hits）
- 检索结果质量

**输出文件**:
- 日志: `test_hyperamy_quick_validation.log`
- 结果: `outputs/three_methods_comparison_monte_cristo/hyperamy_validation_results.json`

### 步骤2: 重新运行完整实验（50个查询）

**目的**: 验证修复后的完整实验效果

**执行方式**:
```bash
# 在实验服务器上运行
ssh hyperamy-server
cd /public/jiangh/HyperAmy

# 确保使用修复后的脚本
git pull origin master

# 运行完整实验（已使用修复后的代码）
source /opt/conda/etc/profile.d/conda.sh
conda activate PyTorch-2.4.1
python -u test/test_three_methods_comparison_monte_cristo.py > test_three_methods_comparison_monte_cristo_new.log 2>&1 &
```

**预期结果**:
- 所有三种方法在所有查询上都成功运行
- HyperAmy的Recall@K > 0%
- 生成完整的结果文件

**验证指标**:
- 三种方法的Recall@K对比
- HyperAmy的检索命中率
- 检索结果质量分析

**输出文件**:
- 日志: `test_three_methods_comparison_monte_cristo_new.log`
- 结果: `outputs/three_methods_comparison_monte_cristo/comparison_results_new.json`

### 步骤3: 对比分析

**目的**: 分析修复前后的差异和HyperAmy的实际表现

**对比内容**:

1. **修复前后结果对比**:
   - 修复前: HyperAmy Recall@K = 0%
   - 修复后: HyperAmy Recall@K = ?
   - 分析差异原因

2. **三种方法性能对比**:
   - HippoRAG vs Fusion vs HyperAmy
   - 在不同Recall@K指标上的表现
   - 优缺点分析

3. **HyperAmy特性分析**:
   - 纯情绪检索的效果
   - 与语义检索的差异
   - 适用场景分析

**分析脚本**:
- 创建对比分析脚本（待实现）

**输出文档**:
- `docs/HYPERAMY_VALIDATION_RESULTS.md` - 验证结果报告
- `docs/HYPERAMY_COMPARISON_ANALYSIS.md` - 对比分析报告

## 时间估计

- **步骤1**（小规模测试）: 约 10-20 分钟
- **步骤2**（完整实验）: 约 14-15 小时
- **步骤3**（对比分析）: 约 30-60 分钟

## 成功标准

### 步骤1成功标准:
- ✅ Recall@5 > 0%
- ✅ 至少有几个查询命中gold_chunk_id
- ✅ 检索结果不为空

### 步骤2成功标准:
- ✅ 所有方法成功运行（100%可用率）
- ✅ HyperAmy Recall@5 > 0%
- ✅ 结果文件正确生成

### 步骤3成功标准:
- ✅ 完成修复前后对比分析
- ✅ 完成三种方法性能对比
- ✅ 生成详细的验证报告

## 后续工作

如果验证成功：
1. 更新实验报告，包含修复后的HyperAmy结果
2. 分析HyperAmy在纯情绪检索上的实际表现
3. 给出三种方法的使用建议

如果验证失败：
1. 进一步诊断问题
2. 检查修复是否完整
3. 可能需要额外的修复

---

**创建时间**: 2026-01-09

