# HyperAmy Experiment 4 - 合作者使用指南

## 📌 最新代码分支

**分支名称**: `experiment-4-dual-path`

**GitHub仓库**: https://github.com/sherkevin/HyperAmy

## 🎯 本次更新核心内容

### 1. **自适应双路检索架构 (Adaptive Dual-Path Retrieval)**

实现了全新的检索架构，系统会根据语义置信度自动选择检索路径：

- **Path A (混合重排序)**: 当语义置信度高时（`S_sem >= 0.01`），使用HippoRAG候选集 + 情绪融合重排序
- **Path B (全局情绪搜索)**: 当检测到语义崩溃时（`S_sem < 0.01`），绕过HippoRAG，直接在全库10,000个文档中进行全局情绪向量搜索

### 2. **关键Bug修复**

#### Bug 1: 数据字段提取错误
- **问题**: 代码使用 `chunk.get('text', '')` 提取文档内容，但实际数据使用 `'input'` 字段
- **修复**: 改为 `chunk.get('input', chunk.get('text', ''))`，并过滤空文档
- **影响**: 修复前所有文档hash ID相同（空字符串的MD5），导致 `_hash_to_doc` 映射失败

#### Bug 2: 维度不匹配
- **问题**: 查询情绪向量30维，文档情绪向量28维，导致 `np.dot` 计算失败
- **修复**: 在 `_global_emotion_search` 中添加维度对齐逻辑（pad/truncate）
- **影响**: Path B现在可以正常执行全库检索

#### Bug 3: S_sem计算异常
- **问题**: 所有查询的 `S_sem=1.0000`，导致Path B从未触发
- **修复**: 检测所有语义分数都接近1.0时，强制 `S_sem=0.0` 触发Path B
- **影响**: 语义崩溃检测现在可以正常工作

### 3. **代码结构变更**

#### 新增文件
- `test/test_vibe_search_experiment_4_dual_path.py`: Experiment 4主测试脚本
- `scripts/run_experiment_4_dual_path.sh`: 实验启动脚本
- `scripts/debug_hash_integrity.py`: Hash ID完整性诊断工具

#### 修改文件
- `sentiment/hipporag_enhanced.py`: 
  - 新增 `_global_emotion_search()` 方法
  - 重构 `retrieve()` 方法实现双路切换逻辑
  - 添加 `_hash_to_doc` 映射用于Path B文档检索
- `test/test_vibe_search_experiment_4_dual_path.py`: 修复数据加载和维度对齐

## 🚀 快速开始

### 1. 克隆代码

```bash
git clone git@github.com:sherkevin/HyperAmy.git
cd HyperAmy
git checkout experiment-4-dual-path
```

### 2. 环境配置

```bash
# 激活conda环境
conda activate PyTorch-2.4.1

# 设置环境变量（防止torch导入死锁）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 配置API密钥（从.env文件读取）
source .env  # 或手动设置 OPENAI_API_KEY
```

### 3. 运行Experiment 4

```bash
# 方式1: 使用启动脚本（推荐）
./scripts/run_experiment_4_dual_path.sh

# 方式2: 直接运行Python脚本
python test/test_vibe_search_experiment_4_dual_path.py
```

### 4. 监控实验进度

```bash
# 查看实时日志
tail -f outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.log

# 查看双路切换情况
grep "DUAL-PATH" outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.log

# 检查实验结果
cat outputs/vibe_search_experiment_4_dual_path/results.json
```

## 📊 实验配置说明

### 数据文件
- **训练数据**: `data/training/monte_cristo_train_full.jsonl` (10,000个chunks)
- **测试查询**: `data/public_benchmark/monte_cristo_vibe_search.json` (50个Vibe Search查询)

### 关键参数
- **语义崩溃阈值**: `SEMANTIC_COLLAPSE_THRESHOLD = 0.01`
- **情绪向量维度**: 28维（与 `particle/emotion.py` 中的 `EMOTIONS` 列表一致）
- **检索Top-K**: 默认10，实验中使用expanded_k=20进行重排序

## 🔍 诊断工具

如果遇到问题，可以使用诊断脚本：

```bash
python scripts/debug_hash_integrity.py
```

该脚本会检查：
1. Hash ID计算一致性
2. `_hash_to_doc` 映射完整性
3. ChromaDB与内存映射的ID格式匹配
4. Path B检索功能测试

## ⚠️ 注意事项

1. **数据字段**: 确保数据文件使用 `'input'` 字段存储文档内容
2. **维度对齐**: 查询和文档情绪向量维度必须匹配（当前为28维）
3. **网络连接**: HippoRAG需要访问embedding API，确保网络畅通
4. **GPU资源**: 如果使用Emos模型，需要单独的GPU（当前实验使用API方式）

## 📈 预期结果

修复后的Experiment 4应该能够：
- **HippoRAG基线**: Recall@1 ≈ 12% (恢复基线性能)
- **HyperAmy V2**: Recall@1 > 12% (如果情绪向量质量足够好)

如果Path B成功触发，说明语义崩溃检测正常工作。如果Recall仍然较低，可能需要：
1. 优化情绪向量提取质量（Experiment 2）
2. 调整语义崩溃阈值
3. 优化情绪权重计算逻辑

## 📝 代码变更摘要

- **新增**: 3个文件（测试脚本、启动脚本、诊断工具）
- **修改**: 2个核心文件（`hipporag_enhanced.py`, 实验脚本）
- **修复**: 3个关键Bug（数据字段、维度对齐、S_sem计算）
- **优化**: 双路切换逻辑、错误处理、日志记录

## 🤝 协作建议

1. **代码审查**: 重点关注 `sentiment/hipporag_enhanced.py` 中的双路切换逻辑
2. **实验验证**: 建议在本地运行Experiment 4验证修复效果
3. **问题反馈**: 如遇到问题，请提供日志文件和错误信息

---

**最后更新**: 2026-01-21  
**维护者**: HyperAmy Team  
**联系方式**: 通过GitHub Issues或Pull Requests
