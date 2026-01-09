# HyperAmy修复验证执行指南

**日期**: 2026-01-09  
**目的**: 验证HyperAmy检索修复是否有效

## 前置条件

✅ **已完成**:
- HyperAmy索引已构建（9,735个点）
- 修复后的代码已提交到GitHub
- 验证脚本已创建

## 验证步骤

### 步骤1: 同步验证脚本到服务器

在本地执行（从项目根目录）:
```bash
# 同步验证脚本
rsync -avz -e "ssh -p 1066" \
  test/test_hyperamy_quick_validation.py \
  scripts/run_hyperamy_validation.sh \
  scripts/start_validation_experiment.sh \
  jiangh@10.103.92.120:/public/jiangh/HyperAmy/

# 或者如果rsync不可用，使用scp
scp -P 1066 \
  test/test_hyperamy_quick_validation.py \
  scripts/run_hyperamy_validation.sh \
  scripts/start_validation_experiment.sh \
  jiangh@10.103.92.120:/public/jiangh/HyperAmy/
```

在服务器上设置执行权限:
```bash
ssh jiangh@10.103.92.120 -p 1066
cd /public/jiangh/HyperAmy
chmod +x test/test_hyperamy_quick_validation.py
chmod +x scripts/run_hyperamy_validation.sh
chmod +x scripts/start_validation_experiment.sh
```

### 步骤2: 运行小规模测试（10个查询）

**选项A: 前台运行（推荐用于调试）**
```bash
ssh jiangh@10.103.92.120 -p 1066
cd /public/jiangh/HyperAmy
source /opt/conda/etc/profile.d/conda.sh
conda activate PyTorch-2.4.1
python -u test/test_hyperamy_quick_validation.py
```

**选项B: 后台运行（推荐用于长时间运行）**
```bash
ssh jiangh@10.103.92.120 -p 1066
cd /public/jiangh/HyperAmy
bash scripts/start_validation_experiment.sh
```

**选项C: 使用nohup（推荐用于远程断开）**
```bash
ssh jiangh@10.103.92.120 -p 1066
cd /public/jiangh/HyperAmy
source /opt/conda/etc/profile.d/conda.sh
conda activate PyTorch-2.4.1
nohup python -u test/test_hyperamy_quick_validation.py > test_hyperamy_quick_validation.log 2>&1 &
echo $! > hyperamy_validation.pid
```

### 步骤3: 查看验证结果

**实时查看日志**:
```bash
tail -f /public/jiangh/HyperAmy/test_hyperamy_quick_validation.log
```

**查看结果文件**:
```bash
cat /public/jiangh/HyperAmy/outputs/three_methods_comparison_monte_cristo/hyperamy_validation_results.json
```

**检查进程状态**:
```bash
ps aux | grep test_hyperamy_quick_validation
```

### 步骤4: 分析验证结果

验证成功标准:
- ✅ Recall@5 > 0%
- ✅ 至少有几个查询命中gold_chunk_id
- ✅ 检索结果不为空

如果验证成功，继续步骤5；如果失败，检查日志并诊断问题。

### 步骤5: 重新运行完整实验（50个查询）

**确保使用修复后的代码**:
```bash
ssh jiangh@10.103.92.120 -p 1066
cd /public/jiangh/HyperAmy
git pull origin master  # 确保使用最新的修复代码
```

**运行完整实验**:
```bash
cd /public/jiangh/HyperAmy
source /opt/conda/etc/profile.d/conda.sh
conda activate PyTorch-2.4.1
nohup python -u test/test_three_methods_comparison_monte_cristo.py > test_three_methods_comparison_monte_cristo_new.log 2>&1 &
echo $! > three_methods_experiment_new.pid
```

**预计耗时**: 约14-15小时

**监控进度**:
```bash
tail -f /public/jiangh/HyperAmy/test_three_methods_comparison_monte_cristo_new.log
```

### 步骤6: 对比分析

**对比修复前后的结果**:

```bash
cd /public/jiangh/HyperAmy
python3 << 'PYEOF'
import json

# 读取修复前的结果
with open('outputs/three_methods_comparison_monte_cristo/comparison_results.json', 'r') as f:
    results_before = json.load(f)

# 读取修复后的结果
with open('outputs/three_methods_comparison_monte_cristo/comparison_results_new.json', 'r') as f:
    results_after = json.load(f)

print("=" * 80)
print("修复前后对比分析")
print("=" * 80)
print()

# 计算修复前HyperAmy的hit率
hyperamy_hits_before = sum(1 for r in results_before if r.get('hyperamy', {}).get('hit', False))
print(f"修复前 - HyperAmy命中数: {hyperamy_hits_before}/{len(results_before)}")
print(f"修复前 - HyperAmy命中率: {hyperamy_hits_before/len(results_before)*100:.1f}%")
print()

# 计算修复后HyperAmy的hit率
hyperamy_hits_after = sum(1 for r in results_after if r.get('hyperamy', {}).get('hit', False))
print(f"修复后 - HyperAmy命中数: {hyperamy_hits_after}/{len(results_after)}")
print(f"修复后 - HyperAmy命中率: {hyperamy_hits_after/len(results_after)*100:.1f}%")
print()

if hyperamy_hits_after > hyperamy_hits_before:
    print("✅ 修复成功！HyperAmy检索性能有所提升")
else:
    print("⚠️  需要进一步分析")
PYEOF
```

## 预期结果

### 小规模测试（10个查询）预期:
- 运行时间: 10-20分钟
- Recall@5: 应该 > 0%（至少1-2个查询命中）
- 输出文件: `hyperamy_validation_results.json`

### 完整实验（50个查询）预期:
- 运行时间: 14-15小时
- 所有方法成功运行: 100%
- HyperAmy Recall@5: 应该 > 0%
- 输出文件: `comparison_results_new.json`

## 故障排除

### 问题1: 脚本找不到
**解决方法**: 
```bash
# 检查文件是否存在
ls -lh /public/jiangh/HyperAmy/test/test_hyperamy_quick_validation.py

# 如果不存在，手动同步或从GitHub拉取
cd /public/jiangh/HyperAmy
git pull origin master
```

### 问题2: 存储目录不存在
**解决方法**:
```bash
# 检查存储目录
ls -ld /public/jiangh/HyperAmy/outputs/three_methods_comparison_monte_cristo/hyperamy_db

# 如果不存在，需要先运行索引脚本
# （此时应该已经存在，因为之前实验已完成）
```

### 问题3: 依赖包缺失
**解决方法**:
```bash
cd /public/jiangh/HyperAmy
source /opt/conda/etc/profile.d/conda.sh
conda activate PyTorch-2.4.1
pip install -r requirements.txt
```

## 相关文档

- [验证计划](./VALIDATION_PLAN.md) - 详细的验证计划
- [修复总结](./HYPERAMY_FIX_SUMMARY.md) - HyperAmy修复详细说明
- [实验结果分析](./EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS.md) - 完整实验分析

---

**创建时间**: 2026-01-09  
**最后更新**: 2026-01-09

