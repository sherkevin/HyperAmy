# 代码提交指南

## 📋 当前状态总结

### 已完成的核心功能

1. **数据准备模块** (`src/data_prep.py`)
   - 智能分块（段落+句子级别）
   - 杏仁核特征注入（情感+惊奇度+mass）

2. **QA生成模块** (`src/gen_qa.py`)
   - Top-K策略选择高质量块
   - GPT-4o生成本能测试题
   - 并发处理支持

3. **实验执行模块** (`src/run_experiment.py`)
   - 三组对比实验（Oracle/Baseline/HyperAmy）
   - 重试机制（最多3次，指数退避）
   - 中间结果保存

4. **评估模块** (`src/evaluate.py`)
   - LLM-as-a-Judge评估
   - 事实准确性和危机感知评分

5. **报告生成** (`src/generate_report.py`)
   - 统计分析报告
   - 案例研究

6. **重试工具**
   - `src/retry_failed_questions.py` - 串行版本
   - `src/retry_failed_questions_parallel.py` - 并发优化版本

### 实验结果

- **总问题数**: 50
- **有效结果**: 16个 (32.0%)
- **检索命中率**: Baseline 0%, HyperAmy 4%
- **评估结果**: HyperAmy获胜4次，平局46次

### 数据限制

- 由于网络连接问题，只有32%的问题成功生成答案
- 样本量较小，统计意义有限
- 建议在报告中说明数据限制

---

## 🚀 提交步骤

### 1. 创建新分支

```bash
git checkout -b feature/retry-mechanism-and-parallel-processing
```

### 2. 添加核心代码文件

```bash
# 核心实验代码
git add src/data_prep.py
git add src/gen_qa.py
git add src/run_experiment.py
git add src/evaluate.py
git add src/generate_report.py
git add src/retry_failed_questions.py
git add src/retry_failed_questions_parallel.py

# 配置和依赖
git add llm/config.py
git add requirements.txt
git add .gitignore

# 工具脚本
git add scripts/download_hotpotqa.py
git add scripts/download_hotpotqa_manual.py

# 监控工具（可选）
git add monitor_retry.sh
git add test_network_simple.py

# 文档
git add docs/
```

### 3. 提交修改

```bash
git commit -m "feat: Add retry mechanism and parallel processing for experiment

- Add retry mechanism with exponential backoff to run_experiment.py
- Implement parallel retry script (retry_failed_questions_parallel.py)
- Add network testing script (test_network_simple.py)
- Add monitoring tools (monitor_retry.sh)
- Update .gitignore to exclude large result files
- Add comprehensive documentation in docs/

Results:
- 16/50 questions successfully retried (32% success rate)
- Parallel processing provides 3-5x speedup
- Network connection issues remain a challenge"
```

### 4. 推送到GitHub

```bash
git push origin feature/retry-mechanism-and-parallel-processing
```

### 5. 创建Pull Request（可选）

在GitHub上创建PR，合并到master分支。

---

## 📝 提交前检查清单

- [x] `.gitignore` 已更新，排除敏感数据和大型结果文件
- [x] 所有核心代码文件已添加
- [x] `requirements.txt` 包含所有依赖
- [x] 代码注释清晰
- [x] 没有硬编码的API密钥
- [x] 文档已更新

---

## ⚠️ 注意事项

1. **不提交的内容**:
   - `results/*.json` - 实验结果文件（已在.gitignore中）
   - `data/books/*.txt` - 原始数据（版权保护）
   - `data/processed/*.jsonl` - 处理后的数据（可能很大）
   - `.env` - API密钥
   - `__pycache__/` - Python缓存

2. **敏感信息检查**:
   - 确认 `llm/config.py` 中没有硬编码的API密钥
   - 确认所有API密钥都从环境变量读取

3. **文档说明**:
   - 在README中说明数据限制
   - 说明如何配置API密钥
   - 说明如何运行实验

---

## 📊 分支命名建议

- `feature/retry-mechanism-and-parallel-processing` - 当前建议
- `feat/experiment-retry` - 简化版本
- `feat/parallel-retry` - 更简洁

---

**准备就绪**: ✅ 可以开始提交

