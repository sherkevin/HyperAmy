# HyperAmy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HyperAmy is an emotion-enhanced RAG framework built on top of [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), integrating emotion analysis capabilities to enable LLMs to understand and leverage emotional context in retrieval-augmented generation tasks.

## Features

- 🧠 **Emotion Analysis**: Extract and quantify emotional content from text
- 🔍 **Hyperbolic Retrieval**: Poincaré ball model for efficient semantic search
- 📊 **Emotion Vectors**: High-dimensional emotion vectors based on affective computing
- 🔄 **Particle Memory**: Time-evolving particle system for memory representation
- 💾 **Persistent Storage**: ChromaDB-based storage with Parquet format
- 🎯 **Multiple Workflow Options**:
  - **Amygdala**: Emotion-enhanced retrieval with particle memory
  - **HippoRAG**: Graph-based RAG with knowledge graph reasoning
  - **Fusion**: Hybrid approaches combining both systems

---

## Installation

### Prerequisites

- Python 3.10+ (recommended: 3.10.18)
- Conda (recommended for environment management)

### Setup

```bash
# Install dependencies
uv sync
# or
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the `llm/` directory:

```bash
API_KEY=your_api_key_here
BASE_URL=https://llmapi.paratera.com/v1
```

---

## Quick Start

HyperAmy 提供三种 workflow 方案，满足不同的检索需求：

### Workflow 1: Amygdala - 情感增强检索

基于粒子记忆和情感向量的检索系统，擅长处理带有情感色彩的对话和文本。

```python
from workflow import Amygdala

# 初始化
amygdala = Amygdala(
    save_dir="./amygdala_db",
    particle_collection_name="particles",
    conversation_namespace="conversations"
)

# 添加对话
result = amygdala.add("I love Python programming! It makes me feel productive.")
print(f"Added {result['particle_count']} particles")

# 检索相关对话片段
results = amygdala.retrieval(
    query_text="programming languages",
    retrieval_mode="chunk",  # 或 "particle"
    top_k=3
)

for result in results:
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
    print(f"Particles: {result['particle_count']}")
```

**适用场景**：
- 对话历史检索
- 情感分析相关的文本检索
- 需要理解情感上下文的场景

### Workflow 2: HippoRAG - 知识图谱检索

基于知识图谱的 RAG 系统，通过 OpenIE 提取实体和关系，构建知识图谱进行推理检索。

```python
from workflow import HippoRAGWrapper

# 初始化
hipporag = HippoRAGWrapper(
    save_dir="./hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    embedding_model_name="GLM-Embedding-2"
)

# 添加文档块
chunks = [
    "Python is a high-level programming language.",
    "JavaScript is widely used for web development."
]
result = hipporag.add(chunks)
print(f"Indexed {result['total_indexed']} chunks")

# 检索相关文档
results = hipporag.retrieve(
    query="What programming languages are mentioned?",
    top_k=2
)

for result in results:
    print(f"Rank {result['rank']}: {result['text']}")
    print(f"Score: {result['score']:.4f}")

# 或使用 RAG 问答
qa_result = hipporag.qa(query="Tell me about Python", top_k=3)
print(f"Answer: {qa_result['answer']}")
```

**适用场景**：
- 文档问答 (QA)
- 知识图谱推理
- 事实性检索
- 需要 OpenIE 提取实体关系的场景

### Workflow 3: Fusion - 融合检索

结合 Amygdala 和 HippoRAG 的优势，提供更强的检索能力。

#### 方案 A: FusionRetriever（级联/并行融合）

```python
from workflow import FusionRetriever

# 初始化融合检索器
fusion = FusionRetriever(
    amygdala_save_dir="./fusion_amygdala_db",
    hipporag_save_dir="./fusion_hipporag_db"
)

# 添加数据（同时添加到两个系统）
chunks = ["Your document chunks..."]
result = fusion.add(chunks)

# 级联检索：HippoRAG 快速筛选 → Amygdala 深度精排
results = fusion.retrieve(
    query="your query",
    hipporag_top_k=20,  # HippoRAG 返回 20 个候选
    amygdala_top_k=5,   # Amygdala 选出 5 个
    mode="cascade"      # 可选: "cascade", "parallel", "hipporag_only", "amygdala_only"
)

for result in results:
    print(f"Rank: {result['rank']}")
    print(f"Text: {result['text']}")
    print(f"HippoRAG Score: {result['hipporag_score']:.4f}")
    print(f"Amygdala Score: {result['amygdala_score']:.4f}")
```

#### 方案 B: GraphFusionRetriever（图谱级融合）

```python
from workflow import GraphFusionRetriever

# 初始化
fusion = GraphFusionRetriever(
    amygdala_save_dir="./graph_fusion_amygdala_db",
    hipporag_save_dir="./graph_fusion_hipporag_db"
)

# 添加数据
chunks = ["Your document chunks..."]
fusion.add(chunks)

# 图谱融合检索：在 HippoRAG 图谱中融合情感信号
results = fusion.retrieve(
    query="your query",
    top_k=5,
    emotion_weight=0.3,    # Amygdala 情感权重
    semantic_weight=0.5,   # HippoRAG 语义权重
    fact_weight=0.2        # HippoRAG fact 权重
)

for result in results:
    print(f"Rank: {result['rank']}")
    print(f"Text: {result['text']}")
    print(f"PPR Score: {result['score']:.4f}")
```

**Fusion 适用场景**：
- 需要同时利用语义和情感信息
- 对检索质量要求高的场景
- 复杂查询需要多路召回

### Workflow 选择指南

| Workflow | 优势 | 劣势 | 推荐使用场景 |
|----------|------|------|--------------|
| **Amygdala** | 情感感知，适合对话检索 | 依赖实体抽取 | 对话系统、情感分析 |
| **HippoRAG** | 知识推理，适合事实检索 | 无情感感知 | 文档问答、知识检索 |
| **FusionRetriever** | 速度快，兼顾两者 | 存储开销大 | 通用检索场景 |
| **GraphFusionRetriever** | 融合度高，效果最好 | 实现复杂 | 高质量要求场景 |

---

## Project Structure

```
HyperAmy/
├── workflow/               # 工作流模块（高级接口）
│   ├── amygdala.py        # Amygdala 工作流：情感增强检索
│   ├── hipporag_wrapper.py # HippoRAG 工作流：知识图谱检索
│   ├── fusion_retrieval.py # FusionRetriever：级联/并行融合
│   └── graph_fusion_retrieval.py # GraphFusionRetriever：图谱级融合
│
├── particle/              # 粒子模块
│   ├── __init__.py
│   ├── emotion_v2.py      # EmotionV2（情感提取）
│   ├── emotion_cache.py   # 情感缓存
│   ├── speed.py           # 速度计算
│   └── temperature.py     # 温度计算
│
├── poincare/              # 双曲空间模块
│   ├── __init__.py
│   ├── types.py           # 数据类型
│   ├── physics.py         # 物理计算（TimePhysics, ParticleProjector）
│   ├── storage.py         # 存储（HyperAmyStorage）
│   ├── retrieval.py       # 检索（HyperAmyRetrieval）
│   └── linking.py         # 链接构建
│
├── llm/                   # LLM 客户端
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   └── completion_client.py # LLM 客户端
│
├── prompts/               # 提示模板
│   └── templates/
│
├── utils/                 # 工具模块
│   ├── sentence.py        # 句子生成
│   ├── ner_lightweight.py # 轻量级 NER
│   └── entity.py          # 实体抽取
│
├── ods/                   # 数据库层
│   └── chroma.py          # ChromaDB 封装
│
├── hipporag/              # HippoRAG 框架（外部依赖）
│
├── test/                  # 测试文件
│   ├── test_amygdala.py           # Amygdala 测试
│   ├── test_hipporag_wrapper.py   # HippoRAG 测试
│   ├── test_fusion_retrieval.py   # FusionRetriever 测试
│   └── test_graph_fusion_*.py     # GraphFusionRetriever 测试
│
└── README.md              # 本文档
```

---

## Workflow API Reference

### Amygdala API

**初始化**

```python
from workflow import Amygdala

amygdala = Amygdala(
    save_dir="./db",                      # 数据库保存路径
    particle_collection_name="particles", # 粒子集合名称
    conversation_namespace="conversations", # 对话命名空间
    embedding_model=None,                  # 嵌入模型（None 使用默认）
    auto_link_particles=True,             # 是否自动链接粒子
    link_distance_threshold=1.5,          # 邻域链接距离阈值
    link_top_k=None                        # 每个粒子的最大邻域数
)
```

**添加文本**

```python
result = amygdala.add(conversation)
# Returns:
# {
#     'conversation_id': str,
#     'particles': List[ParticleEntity],
#     'particle_count': int,
#     'relationship_map': Dict[str, str]
# }
```

**检索**

```python
# Particle 模式 - 返回粒子
particles = amygdala.retrieval(
    query_text="your query",
    retrieval_mode="particle",
    top_k=10,
    cone_width=50,
    max_neighbors=20
)

# Chunk 模式 - 返回对话片段
chunks = amygdala.retrieval(
    query_text="your query",
    retrieval_mode="chunk",
    top_k=5
)
```

**参数说明**：
- `query_text` (str): 查询文本
- `retrieval_mode` (str): `"particle"` 或 `"chunk"`
- `top_k` (int): 返回结果数量
- `cone_width` (int): 锥体搜索宽度（50-100）
- `max_neighbors` (int): 邻域扩展最大节点数
- `neighbor_penalty` (float): 邻居惩罚系数（默认 1.1）

**Chunk 得分计算**：
```
chunk_score = sum((total_particles - position) for each particle in chunk)
```
位置靠前的粒子贡献更大，包含更多靠前粒子的 chunk 得分更高。

### HippoRAG API

**初始化**

```python
from workflow import HippoRAGWrapper

hipporag = HippoRAGWrapper(
    save_dir="./hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    embedding_model_name="GLM-Embedding-2",
    llm_base_url=None,  # 可选，覆盖默认 URL
    embedding_base_url=None  # 可选，覆盖默认 URL
)
```

**添加文档**

```python
result = hipporag.add(chunks)
# Returns:
# {
#     'chunk_count': int,
#     'chunk_ids': List[str],
#     'total_indexed': int
# }
```

**检索**

```python
# 标准检索（使用图谱）
results = hipporag.retrieve(
    query="your query",
    top_k=5,
    return_scores=True
)

# DPR 检索（不使用图谱，更快但精度较低）
results = hipporag.retrieve_dpr(
    query="your query",
    top_k=5
)

# RAG 问答
qa_result = hipporag.qa(
    query="your question",
    top_k=5
)
# Returns:
# {
#     'answer': str,
#     'retrieved_chunks': List[Dict],
#     'messages': List,
#     'metadata': Dict
# }
```

**其他方法**

```python
# 删除文档
hipporag.delete(chunks)

# 获取统计信息
stats = hipporag.get_stats()
# Returns:
# {
#     'total_indexed': int,
#     'graph_nodes': int,
#     'graph_edges': int,
#     'entities': int,
#     'facts': int
# }

# 清空索引
hipporag.clear()
```

### FusionRetriever API

**初始化**

```python
from workflow import FusionRetriever

fusion = FusionRetriever(
    amygdala_save_dir="./fusion_amygdala_db",
    hipporag_save_dir="./fusion_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    embedding_model_name="GLM-Embedding-2",
    auto_link_particles=False
)
```

**添加数据**

```python
result = fusion.add(chunks)
# Returns:
# {
#     'amygdala_count': int,
#     'hipporag_count': int,
#     'total_chunks': int
# }
```

**融合检索**

```python
# 级联检索（推荐）
results = fusion.retrieve(
    query="your query",
    hipporag_top_k=20,  # HippoRAG 返回候选数
    amygdala_top_k=5,   # 最终返回数
    mode="cascade"      # 检索模式
)

# 可选模式：
# - "cascade": 级联检索（默认）
# - "parallel": 并行检索 + 分数融合
# - "hipporag_only": 仅 HippoRAG
# - "amygdala_only": 仅 Amygdala
```

**返回结果格式**：
```python
{
    'rank': int,
    'text': str,
    'hipporag_score': float,
    'amygdala_score': float,
    'fusion_score': float
}
```

### GraphFusionRetriever API

**初始化**

```python
from workflow import GraphFusionRetriever

fusion = GraphFusionRetriever(
    amygdala_save_dir="./graph_fusion_amygdala_db",
    hipporag_save_dir="./graph_fusion_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    embedding_model_name="GLM-Embedding-2"
)
```

**融合检索**

```python
results = fusion.retrieve(
    query="your query",
    top_k=5,
    emotion_weight=0.3,    # Amygdala 情绪权重
    semantic_weight=0.5,   # HippoRAG 语义权重
    fact_weight=0.2,       # HippoRAG fact 权重
    linking_top_k=20,      # HippoRAG 链接 top_k
    passage_node_weight=0.05  # passage 节点权重
)
```

**检索流程**：
1. 从 query 中抽取实体
2. HippoRAG 语义扩展实体
3. Amygdala 情绪扩展实体
4. HippoRAG fact 提取实体
5. 融合实体权重
6. 运行 PPR 传播
7. 返回排序后的 chunks

---

## Advanced Usage

### 粒子创建 -> 存储 -> 查询完整流程

如果你想深入理解底层机制，可以参考以下流程：

#### Step 1: 创建粒子

```python
from particle import Particle

particle = Particle()
particles = particle.process(
    text="I enjoy coding with Python",
    text_id="doc1"
)
# particles: List[ParticleEntity]
```

#### Step 2: 存储到双曲空间

```python
from poincare import HyperAmyStorage

storage = HyperAmyStorage(
    persist_path="./db",
    collection_name="particles"
)

# 批量存储
storage.upsert_entities(entities=particles)
```

#### Step 3: 查询粒子

```python
from poincare import HyperAmyRetrieval, ParticleProjector

projector = ParticleProjector(curvature=1.0, scaling_factor=2.0)
retrieval = HyperAmyRetrieval(storage, projector)

results = retrieval.search(
    query_entity=query_particle,
    top_k=10,
    cone_width=50
)
```

**检索流程**（四步混合检索）:
1. **锥体锁定**: 使用向量相似度快速圈定方向一致的粒子
2. **壳层筛选**: 计算真实的双曲距离进行精排
3. **邻域激活**: 从 Top-K 点出发，扩展其邻居节点
4. **汇总排序**: 混合直接检索点和邻居点，最终排序返回

**推荐**：对于大多数使用场景，直接使用 workflow 模块的高级接口（Amygdala/HippoRAG/Fusion）即可，无需手动处理这些底层细节。

---

## Module Documentation

### Workflow Modules

#### workflow.amygdala

**Amygdala** - 情感增强检索工作流

```python
from workflow import Amygdala

# Initialize
amygdala = Amygdala(save_dir="./db")

# Add text
result = amygdala.add("Your text here")

# Retrieve
results = amygdala.retrieval("Your query", retrieval_mode="chunk")
```

#### workflow.hipporag_wrapper

**HippoRAGWrapper** - HippoRAG 简洁接口

```python
from workflow import HippoRAGWrapper

# Initialize
hipporag = HippoRAGWrapper(save_dir="./db")

# Add documents
result = hipporag.add(chunks)

# Retrieve
results = hipporag.retrieve("Your query", top_k=5)

# QA
qa_result = hipporag.qa("Your question", top_k=5)
```

#### workflow.fusion_retrieval

**FusionRetriever** - 级联/并行融合检索

```python
from workflow import FusionRetriever

# Initialize
fusion = FusionRetriever(
    amygdala_save_dir="./amygdala_db",
    hipporag_save_dir="./hipporag_db"
)

# Add data
result = fusion.add(chunks)

# Retrieve (cascade mode)
results = fusion.retrieve(
    query="your query",
    hipporag_top_k=20,
    amygdala_top_k=5,
    mode="cascade"
)
```

#### workflow.graph_fusion_retrieval

**GraphFusionRetriever** - 图谱级融合检索

```python
from workflow import GraphFusionRetriever

# Initialize
fusion = GraphFusionRetriever(
    amygdala_save_dir="./amygdala_db",
    hipporag_save_dir="./hipporag_db"
)

# Add data
result = fusion.add(chunks)

# Retrieve with custom weights
results = fusion.retrieve(
    query="your query",
    top_k=5,
    emotion_weight=0.3,
    semantic_weight=0.5,
    fact_weight=0.2
)
```

### Core Modules

#### particle

**Particle** - 粒子处理和生成

```python
from particle import Particle

particle = Particle()
particles = particle.process(text="Your text", text_id="doc1")
```

- `emotion_v2.py`: EmotionV2 - 情感提取和情感描述
- `speed.py`: 速度计算
- `temperature.py`: 温度计算

#### poincare

**双曲空间** - Poincaré 球模型的存储和检索

```python
from poincare import HyperAmyStorage, HyperAmyRetrieval, ParticleProjector

# Storage
storage = HyperAmyStorage(persist_path="./db")
storage.upsert_entities(entities=particles)

# Retrieval
projector = ParticleProjector()
retrieval = HyperAmyRetrieval(storage, projector)
results = retrieval.search(query_entity, top_k=10)
```

- `types.py`: Point, SearchResult 数据类型
- `physics.py`: TimePhysics, ParticleProjector
- `storage.py`: HyperAmyStorage
- `retrieval.py`: HyperAmyRetrieval
- `linking.py`: 链接构建

#### llm

**LLM Client** - 统一的 LLM 接口

```python
from llm import create_client

client = create_client(model_name="DeepSeek-V3.2")
result = client.complete("Your question", mode="normal")
print(result.get_answer_text())
```

---

## Tests

### 测试文件说明

```bash
# Amygdala 工作流测试
python test/test_amygdala.py

# HippoRAG 工作流测试
python test/test_hipporag_wrapper.py

# FusionRetriever 测试
python test/test_fusion_retrieval.py

# GraphFusionRetriever 测试（简化版）
python test/test_fusion_comparison_simple.py

# GraphFusionRetriever 测试（详细版）
python test/test_fusion_comparison_detailed.py

# GraphFusionRetriever 测试（快速版）
python test/test_fusion_comparison_quick.py
```

### 运行测试并保存日志

```bash
# 运行测试并保存日志
python test/test_fusion_comparison_simple.py 2>&1 | tee log/test_fusion_simple.log
```

---

---

## Dependencies

### Required

- `requests>=2.32.0`
- `python-dotenv>=1.1.0`
- `numpy>=1.26.0`
- `pandas>=2.0.0`
- `openai>=1.91.0`
- `httpx>=0.28.0`
- `pyarrow>=14.0.0`
- `chromadb>=0.5.0`
- `tenacity>=8.5.0`
- `tqdm>=4.66.0`

### Optional

- `transformers>=4.45.0`
- `sentence-transformers>=2.2.0`
- `torch>=2.0.0`

---

## Key Concepts

### 粒子记忆 (Particle Memory)

HyperAmy 使用"粒子"来表示文本中的关键实体和概念。每个粒子包含：

```python
class ParticleEntity:
    entity_id: str           # 唯一标识
    entity: str              # 实体名称
    text_id: str             # 文本 ID
    emotion_vector: np.ndarray # 情感向量（高维）
    weight: float            # 权重
    speed: float             # 速度/强度
    temperature: float       # 温度/熵
    born: float              # 生成时间
```

**情感向量**：基于情感计算模型，将文本的情感维度编码为高维向量，包含：
- Valence（愉悦度）
- Arousal（激活度）
- Dominance（支配度）
- 以及其他情感维度

### 双曲空间 (Hyperbolic Space)

HyperAmy 使用 Poincaré 球模型进行向量存储和检索：

**优势**：
- 能够更好地表示层级关系
- 相比欧几里得空间，相似概念的距离更近
- 适合表示知识图谱和语义关系

**双曲距离**：
- 距离越小，粒子越相似
- 粒子到自己的距离接近 0
- 相似情绪和强度的粒子距离较小

### 知识图谱检索 (HippoRAG)

基于 HippoRAG 的知识图谱检索机制：

**核心流程**：
1. **OpenIE 提取**：从文本中提取实体和三元组（主语-谓语-宾语）
2. **图谱构建**：构建包含实体节点、事实节点、文档节点的知识图谱
3. **PPR 传播**：使用 Personalized PageRank 在图谱上传播相关性
4. **结果排序**：返回最相关的文档块

**优势**：
- 能够进行多跳推理
- 利用实体关系提升检索质量
- 适合事实性问答

### 融合策略 (Fusion Strategies)

HyperAmy 提供多种融合策略：

#### 1. 级联检索 (Cascade)
```
Query → HippoRAG (Top-K 候选) → Amygdala (精排) → Final Results
```
- 速度快
- HippoRAG 快速缩小范围
- Amygdala 深度精排

#### 2. 并行检索 (Parallel)
```
Query → HippoRAG ─┐
                 ├→ 分数融合 → Final Results
Query → Amygdala ─┘
```
- 两个系统并行工作
- 保留双方信号
- 分数归一化后融合

#### 3. 图谱融合 (Graph Fusion)
```
Query → 实体抽取
         ├→ HippoRAG 语义扩展
         ├→ Amygdala 情绪扩展
         └→ Fact 扩展
         ↓
    融合实体权重 → PPR 传播 → Final Results
```
- 最深度的融合
- 在图谱层面整合情感信号
- 检索质量最高

### 检索模式对比

| 特性 | Amygdala | HippoRAG | FusionRetriever | GraphFusionRetriever |
|------|----------|----------|-----------------|---------------------|
| **情感感知** | ✓ | ✗ | ✓ | ✓ |
| **知识推理** | ✗ | ✓ | ✓ | ✓ |
| **检索速度** | 中 | 快 | 中-快 | 慢 |
| **检索质量** | 中 | 高 | 高 | 最高 |
| **存储开销** | 中 | 中 | 大 | 大 |
| **实现复杂度** | 低 | 低 | 中 | 高 |
| **推荐场景** | 对话检索 | 事实问答 | 通用检索 | 高质量要求 |

---

## Citation

If you use HyperAmy in your research, please cite:

```bibtex
@misc{hyperamy2024,
  title={HyperAmy: Emotion-Enhanced RAG Framework},
  author={HyperAmy Contributors},
  year={2024},
  url={https://github.com/sherkevin/HyperAmy}
}
```

And the base HippoRAG framework:

```bibtex
@inproceedings{gutiérrez2024hipporag,
  title={HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models},
  author={Bernal Jiménez Gutiérrez and Yiheng Shu and Yu Gu and Michihiro Yasunaga and Yu Su},
  booktitle={NeurIPS},
  year={2024}
}
```

---

## License

MIT License

---

**HyperAmy**: Emotion-Enhanced RAG Framework built on HippoRAG
