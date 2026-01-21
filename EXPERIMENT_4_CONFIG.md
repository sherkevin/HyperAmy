# Experiment 4: 详细实验配置说明

## 📋 实验概述

**实验名称**: HyperAmy V2 - Adaptive Dual-Path Retrieval  
**实验目标**: 解决Vibe Search任务中因HippoRAG召回率低导致的性能瓶颈  
**核心策略**: 当检测到语义崩溃时，绕过HippoRAG候选集，直接执行全库情绪向量检索

## 🔧 核心配置参数

### 1. 语义崩溃检测阈值

```python
SEMANTIC_COLLAPSE_THRESHOLD = 0.01
```

- **位置**: `sentiment/hipporag_enhanced.py` 第380行
- **作用**: 当语义置信度 `S_sem < 0.01` 时，触发Path B（全局情绪搜索）
- **计算方式**: `S_sem = max(semantic_scores)`，如果所有分数都是1.0，则强制 `S_sem = 0.0`

### 2. 情绪权重配置

```python
sentiment_weight = 0.4
semantic_weight = 0.6  # 自动计算为 1 - sentiment_weight
```

- **位置**: `test/test_vibe_search_experiment_4_dual_path.py` 第118行
- **默认值**: 0.4（基于Experiment 3的最佳实践）
- **Path A**: 使用动态权重（基于融合策略计算）
- **Path B**: 固定为 `w_emo = 1.0, w_sem = 0.0`

### 3. 融合策略配置

```python
fusion_strategy = FusionStrategy.HARMONIC
normalization_strategy = NormalizationStrategy.NONE
```

- **位置**: `sentiment/hipporag_enhanced.py` 第96-97行
- **融合策略**: HARMONIC（调和平均，基于Experiment 3最佳实践）
- **归一化策略**: NONE（不归一化，避免信息损失）

### 4. 检索参数

```python
# 基础检索Top-K
num_to_retrieve = 5  # 最终返回的文档数

# 扩展检索（用于重排序）
expanded_k = num_to_retrieve * 2  # 检索更多候选用于重排序

# Path B全局检索
top_k = expanded_k  # 全库检索返回的文档数
```

- **位置**: `sentiment/hipporag_enhanced.py` 第383行
- **逻辑**: 
  - 先检索 `expanded_k` 个候选（通常是最终需要的2倍）
  - Path A: 在这 `expanded_k` 个候选中进行重排序
  - Path B: 在全库中检索 `expanded_k` 个文档

### 5. 情绪向量配置

```python
# 情绪向量维度
emotion_dim = 28  # 与 particle/emotion.py 中的 EMOTIONS 列表一致

# 查询情绪向量维度（可能不同）
query_emotion_dim = 30  # LLM API返回的维度

# 维度对齐策略
# 如果 query_dim > doc_dim: 截断到 doc_dim
# 如果 query_dim < doc_dim: 填充0到 doc_dim
```

- **位置**: `sentiment/hipporag_enhanced.py` 的 `_global_emotion_search` 方法
- **处理**: 自动进行维度对齐（pad/truncate）

## 📊 数据集配置

### 训练数据

```python
chunks_file = "data/training/monte_cristo_train_full.jsonl"
```

- **格式**: JSONL，每行一个JSON对象
- **字段**: `{'input': '文档内容', ...}` 或 `{'text': '文档内容', ...}`
- **数量**: 约10,000个chunks
- **用途**: 构建HippoRAG索引和情绪向量库

### 测试数据

```python
vibe_file = "data/public_benchmark/monte_cristo_vibe_search.json"
```

- **格式**: JSON，包含 `{'data': [...]}` 数组
- **字段**: `{'query': '查询文本', 'gold_text': '正确答案', ...}`
- **数量**: 50个Vibe Search查询
- **用途**: 评估检索性能

## 🎯 模型配置

### LLM模型

```python
llm_model_name = DEFAULT_MODEL  # 从 llm/config.py 读取
llm_base_url = BASE_URL  # 从 llm/config.py 读取
```

- **用途**: 提取情绪向量、NER实体提取
- **配置位置**: `llm/config.py`

### Embedding模型

```python
embedding_model_name = DEFAULT_EMBEDDING_MODEL  # 从 llm/config.py 读取
embedding_base_url = API_URL_EMBEDDINGS  # 从 llm/config.py 读取
```

- **用途**: 生成文档和查询的语义嵌入
- **配置位置**: `llm/config.py`

## 🔄 双路切换逻辑

### Path A: Hybrid Re-ranking（混合重排序）

**触发条件**: `S_sem >= 0.01`

**流程**:
1. 使用HippoRAG获取 `expanded_k` 个语义候选
2. 提取每个候选文档的情绪向量
3. 计算查询与候选的情绪相似度（余弦相似度）
4. 使用融合策略（HARMONIC）融合语义分数和情绪分数
5. 重排序并返回Top-K

**权重**: 动态计算（基于融合策略）

### Path B: Global Emotion Search（全局情绪搜索）

**触发条件**: `S_sem < 0.01` 或所有语义分数都是1.0

**流程**:
1. 丢弃HippoRAG的语义候选
2. 直接在全库（10,000个文档）中进行情绪向量搜索
3. 使用余弦相似度计算查询情绪向量与所有文档情绪向量的相似度
4. 返回Top-K个最相似的文档

**权重**: 固定 `w_emo = 1.0, w_sem = 0.0`

## 📝 关键代码位置

### 核心方法

1. **`_global_emotion_search`**: `sentiment/hipporag_enhanced.py` 第301-355行
   - 实现Path B的全库情绪检索
   - 处理维度对齐
   - 使用余弦相似度计算

2. **`retrieve`**: `sentiment/hipporag_enhanced.py` 第359-570行
   - 实现双路切换逻辑
   - 计算S_sem
   - 根据阈值选择Path A或Path B

3. **`index`**: `sentiment/hipporag_enhanced.py` 第188-250行
   - 构建文档索引
   - 提取情绪向量
   - 填充 `_hash_to_doc` 映射

### 实验脚本

- **主脚本**: `test/test_vibe_search_experiment_4_dual_path.py`
- **启动脚本**: `scripts/run_experiment_4_dual_path.sh`

## 🐛 已修复的关键Bug

### Bug 1: 数据字段提取错误

- **问题**: 代码使用 `chunk.get('text', '')`，但数据使用 `'input'` 字段
- **修复**: 改为 `chunk.get('input', chunk.get('text', ''))`
- **位置**: `test/test_vibe_search_experiment_4_dual_path.py` 第125行

### Bug 2: 维度不匹配

- **问题**: 查询30维，文档28维，导致 `np.dot` 失败
- **修复**: 在 `_global_emotion_search` 中添加维度对齐
- **位置**: `sentiment/hipporag_enhanced.py` 第310-325行

### Bug 3: S_sem计算异常

- **问题**: 所有查询的 `S_sem=1.0000`，导致Path B从未触发
- **修复**: 检测所有分数都是1.0时，强制 `S_sem=0.0`
- **位置**: `sentiment/hipporag_enhanced.py` 第442-448行

### Bug 4: `_hash_to_doc` 为空

- **问题**: 文档索引未正确填充，导致Path B无法查找文档
- **修复**: 在 `index` 方法中正确填充 `_hash_to_doc`
- **位置**: `sentiment/hipporag_enhanced.py` 第240-245行

## 📈 评估指标

### Recall@K

```python
recall_at_k = {
    'Recall@1': 0.0,
    'Recall@5': 0.0,
    'Recall@10': 0.0,
    'Recall@20': 0.0
}
```

- **计算方式**: 在Top-K检索结果中，包含正确答案的比例
- **位置**: `test/test_vibe_search_experiment_4_dual_path.py` 第163-185行

### 对比方法

1. **HippoRAG基线**: 纯语义检索，不使用情绪信息
2. **HyperAmy V2**: 自适应双路检索

## 🔍 日志和监控

### 关键日志标记

- `[DUAL-PATH] ⚠️ Semantic Collapse`: Path B触发
- `[DUAL-PATH] ✅ Semantic Signal Strong`: Path A使用
- `Recall@1`, `Recall@5`: 评估指标

### 日志文件

- **实验日志**: `outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.log`
- **结果文件**: `outputs/vibe_search_experiment_4_dual_path/results.json`

## ⚙️ 环境配置

### 必需环境变量

```bash
export OMP_NUM_THREADS=1  # 防止torch导入死锁
export MKL_NUM_THREADS=1
export OPENAI_API_KEY=<your_api_key>
export API_KEY=<your_api_key>
```

### Conda环境

```bash
conda activate PyTorch-2.4.1
```

## 📚 相关文件

- **核心实现**: `sentiment/hipporag_enhanced.py`
- **融合策略**: `sentiment/fusion_strategies.py`
- **情绪提取**: `sentiment/emotion_vector.py`
- **情绪存储**: `sentiment/emotion_store.py`
- **实验脚本**: `test/test_vibe_search_experiment_4_dual_path.py`
- **启动脚本**: `scripts/run_experiment_4_dual_path.sh`
- **使用指南**: `COLLABORATOR_GUIDE.md`

---

**最后更新**: 2026-01-21  
**维护者**: HyperAmy Team
