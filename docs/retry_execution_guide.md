# 重试失败问题执行指南

## ✅ 网络测试结果

**测试通过！** 网络连接稳定，成功率100%。

- ✅ 简单API调用：成功
- ✅ 连续5次调用：5/5成功 (100%)
- ✅ 带上下文的调用：成功

## 📋 执行步骤

### 选项A：在远程服务器上运行（推荐）

远程服务器环境完整，依赖已安装。

#### 1. 同步更新的代码

```bash
# 同步更新的 run_experiment.py 和 retry_failed_questions.py
rsync -avz \
  src/run_experiment.py \
  src/retry_failed_questions.py \
  test_network_simple.py \
  your-user@your-server:/media/data4/jiangh/Amygdala/hyperamy_source/src/
```

#### 2. SSH到远程服务器并运行

```bash
ssh your-user@your-server

# 激活conda环境
source /media/data4/jiangh/conda/etc/profile.d/conda.sh
conda activate Amygdala

# 进入项目目录
cd /media/data4/jiangh/Amygdala/hyperamy_source

# 先测试网络（可选）
python test_network_simple.py

# 运行重试脚本
python src/retry_failed_questions.py \
  --input results/experiment_full.json \
  --output results/experiment_full_retried.json
```

#### 3. 监控进度

```bash
# 查看日志（如果有）
tail -f retry.log

# 或者查看结果文件（会实时更新）
watch -n 5 'wc -l results/experiment_full_retried.json'
```

### 选项B：在本地运行（需要先安装依赖）

如果要在本地运行，需要先安装依赖：

```bash
# 安装依赖
pip install sentence-transformers faiss-cpu torch transformers

# 然后运行
python src/retry_failed_questions.py \
  --input results/experiment_full.json \
  --output results/experiment_full_retried.json
```

## 📊 预期结果

- **输入**: 50个问题，其中39个失败
- **输出**: 更新后的结果文件，失败的问题会被重新运行
- **预期成功率**: 基于网络测试，预计80-90%的问题可以成功重试

## ⏱️ 预计时间

- 每个问题约需10-30秒（包括检索和答案生成）
- 39个失败问题 × 3组（Oracle, Baseline, HyperAmy） = 117次API调用
- 预计总时间：20-60分钟（取决于网络和API响应速度）

## 🔍 验证结果

运行完成后，检查结果：

```python
import json

# 加载结果
with open('results/experiment_full_retried.json', 'r') as f:
    results = json.load(f)

# 统计有效结果
valid = 0
for r in results:
    oracle_ok = '出错' not in r.get('oracle', {}).get('answer', '')
    baseline_ok = '出错' not in r.get('baseline', {}).get('answer', '')
    hyperamy_ok = '出错' not in r.get('hyperamy', {}).get('answer', '')
    if oracle_ok and baseline_ok and hyperamy_ok:
        valid += 1

print(f"有效结果: {valid}/{len(results)} ({100*valid/len(results):.1f}%)")
```

## 🛠️ 故障排除

### 如果仍然有大量失败

1. **检查网络连接**
   ```bash
   python test_network_simple.py
   ```

2. **检查API密钥**
   - 确认 `.env` 文件中的 `API_KEY` 正确
   - 确认 `BASE_URL` 正确

3. **增加重试次数**
   - 修改 `src/run_experiment.py` 中的 `max_retries` 参数
   - 默认是3次，可以增加到5次

4. **分批运行**
   - 可以修改脚本，每次只重试10个问题
   - 避免一次性运行太多导致API限流

## 📝 下一步

重试完成后：

1. **验证数据质量**
   - 检查有效结果数量
   - 如果有效结果 > 40个，可以继续评估

2. **重新评估**
   ```bash
   python src/evaluate.py \
     --input results/experiment_full_retried.json \
     --output results/evaluation_results_retried.json
   ```

3. **生成报告**
   ```bash
   python src/generate_report.py \
     --evaluation results/evaluation_results_retried.json \
     --experiment results/experiment_full_retried.json \
     --output results/analysis_report_retried.md
   ```

---

**状态**: ✅ 网络测试通过，可以运行全量重试

