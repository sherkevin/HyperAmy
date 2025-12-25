# HyperAmy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HyperAmy is an emotion-enhanced RAG framework built on top of [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), integrating emotion analysis capabilities to enable LLMs to understand and leverage emotional context in retrieval-augmented generation tasks.

```
HyperAmy/
├── llm/                    # LLM 客户端模块
│   ├── __init__.py         # 模块导出
│   ├── config.py           # 配置管理（从 .env 读取 API_KEY 和 BASE_URL）
│   ├── completion_client.py # LLM 客户端（支持 normal 和 specific 两种模式）
│   └── README.md           # LLM 模块详细文档
│
├── point_label/            # 点标签模块（情感、记忆深度、温度、惊讶值）
│   ├── __init__.py
│   ├── emotion.py          # 情感向量提取（输入 chunk，输出 emotion vector）
│   ├── labels.py           # 记忆深度和温度计算（emotion vector, memory_depth, temperature）
│   ├── speed.py            # 惊讶值计算（surprise value，基于 token 概率）
│   └── temperature.py     # 温度计算（待实现）
│
├── poincare/               # 双曲空间存储与检索模块
│   ├── __init__.py
│   ├── types.py            # 数据类型定义（Point, SearchResult）
│   ├── physics.py          # 双曲空间物理计算（TimePhysics, ParticleProjector）
│   ├── storage.py          # 双曲空间存储（HyperAmyStorage）
│   ├── retrieval.py        # 双曲空间检索（HyperAmyRetrieval）
│   └── linking.py          # 双曲空间链接构建
│
├── sentiment/              # 情感分析模块（旧版，保留兼容）
│   ├── __init__.py
│   ├── emotion_vector.py   # 情感向量提取
│   ├── emotion_store.py    # 情感向量存储和管理
│   └── hipporag_enhanced.py # HippoRAG 增强版（集成情感分析）
│
├── utils/                  # 工具模块
│   └── extract_entitiy.py  # 实体抽取（基于 HippoRAG 的 OpenIE）
│
└── test/                   # 测试文件
    ├── test_infer.py       # 测试推理和 token 概率分析
    ├── test_completion_client.py # 测试 Completion Client 功能
    ├── test_emotion.py     # 测试情感向量提取
    ├── test_bge.py         # 测试 BGE 嵌入和情感描述
    ├── test_integration.py # 测试 HippoRAG 整合
    ├── test_dataset_integration.py # 测试数据集整合
    ├── test_labels.py     # 测试记忆深度和温度计算
    ├── test_speed.py       # 测试惊讶值计算
    ├── test_entity.py     # 测试实体抽取
    ├── test_poincare.py   # 测试双曲空间存储和检索
    └── test_linking.py    # 测试双曲空间链接
```

HyperAmy extends HippoRAG with emotion-aware capabilities:

- **Emotion-Enhanced Retrieval**: Combines semantic and emotional similarity for more contextually relevant document retrieval
- **Emotion Vector Extraction**: Extracts 28-dimensional emotion vectors from text using LLMs
- **Emotion-Aware RAG**: Integrates emotional understanding into the RAG pipeline for improved answer quality
- **Token-Level Probability Analysis**: Supports detailed token-level probability analysis for LLM outputs

## Features

- 🧠 **Emotion Analysis**: Extract and quantify emotional content from text
- 🔍 **Emotion-Enhanced Retrieval**: Combine semantic and emotional similarity for better retrieval
- 📊 **Emotion Vectors**: 28-dimensional emotion vectors based on Plutchik's emotion wheel
- 🔄 **Seamless Integration**: Built on HippoRAG framework with minimal changes
- 🎯 **Token Probability**: Support for token-level probability analysis
- 💾 **Persistent Storage**: Efficient storage of emotion vectors using Parquet format

## Installation

### Prerequisites

- Python 3.10+ (recommended: 3.10.18)
- Conda (recommended for environment management)

### Setup

#### Option 1: Using Conda (Recommended)

```bash
uv sync
```

#### Option 2: Using pip

```bash
# Ensure Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the `llm/` directory:

```bash
API_KEY=your_api_key_here
BASE_URL=https://llmapi.paratera.com/v1
```

**Note**: 
- The `.env` file should only contain `API_KEY` and `BASE_URL`
- Model names are specified in code, not as environment variables
- Configuration is managed through the `llm.config` module

### Verify Installation

Run the environment check script:

```bash
python scripts/check_environment.py
```

#### 点标签模块测试
You should see:
- ✅ Python version: 3.10.18
- ✅ All required dependencies installed
- ✅ API configuration correct

## Quick Start

# 测试记忆深度和温度计算
python -m test.test_labels

# 测试惊讶值计算
python -m test.test_speed
```

#### 实体抽取测试

```bash
# 测试实体抽取功能
python -m test.test_entity
```

#### 双曲空间模块测试

```bash
# 测试双曲空间存储和检索
python -m test.test_poincare

# 测试双曲空间链接
python -m test.test_linking
```

#### 整合测试

```bash
# 测试 HippoRAG 整合（小样本数据）
python -m test.test_integration

# 测试数据集整合（真实数据集）
python -m test.test_dataset_integration
```

## 主要模块说明

### llm 模块

- **`llm/config.py`**：统一管理 API 配置，从 `.env` 文件读取 `API_KEY` 和 `BASE_URL`
- **`llm/completion_client.py`**：LLM 客户端封装
  - `CompletionClient`：支持 normal 和 specific 两种模式
  - `create_client()`：便捷函数创建客户端
  - `ChatResult`：普通对话结果（normal 模式）
  - `CompletionResult`：带 token 概率的结果（specific 模式）

### point_label 模块

点标签模块提供了多种文本特征提取功能：

- **`point_label/emotion.py`**：情感向量提取
  - `Emotion` 类：输入 chunk，输出 30 维情感向量（归一化）
  - 基于 Plutchik 情绪轮和扩展情绪列表

- **`point_label/labels.py`**：记忆深度和温度计算
  - `Labels` 类：输入 chunk，输出 `LabelsResult`（包含 emotion_vector, memory_depth, temperature）
  - `memory_depth`：记忆深度 = 纯度 × 归一化模长（0~1）
  - `temperature`：温度 = f(纯度, 困惑度)，表示情绪波动程度（仅在 `use_specific=True` 时计算）

- **`point_label/speed.py`**：惊讶值计算
  - `Speed` 类：输入 chunk，输出惊讶值（surprise value）
  - 基于信息论的 surprisal：`surprisal = -log(p)`
  - 支持多种聚合方式：mean（推荐）、sum、max、geometric_mean

### poincare 模块

双曲空间存储与检索模块，实现基于 Poincaré 球的情绪记忆系统：

- **`poincare/types.py`**：数据类型定义
  - `Point`：双曲空间中的点（包含位置、速度、时间等属性）
  - `SearchResult`：检索结果

- **`poincare/physics.py`**：双曲空间物理计算
  - `TimePhysics`：时间物理计算
  - `ParticleProjector`：粒子投影器

- **`poincare/storage.py`**：双曲空间存储
  - `HyperAmyStorage`：基于 ChromaDB 的双曲空间存储

- **`poincare/retrieval.py`**：双曲空间检索
  - `HyperAmyRetrieval`：混合检索（语义检索 + 双曲空间检索）

- **`poincare/linking.py`**：双曲空间链接构建
  - `build_hyperbolic_links`：构建双曲空间链接
  - `update_points_with_links`：更新点的链接信息
  - `auto_link_points`：自动链接点

### utils 模块

- **`utils/extract_entitiy.py`**：实体抽取
  - `Entity` 类：基于 HippoRAG 的 OpenIE 模块
  - `extract_entities()`：提取命名实体
  - `extract_triples()`：提取三元组（实体-关系-实体）
  - `extract_all()`：同时提取实体和三元组

### sentiment 模块（旧版，保留兼容）

- **`sentiment/emotion_vector.py`**：从文本中提取情感向量
- **`sentiment/emotion_store.py`**：情感向量的存储和管理
- **`sentiment/hipporag_enhanced.py`**：增强版 HippoRAG，集成情感分析功能

## 使用示例

### 基本使用
### Basic Usage

#### Using LLM Client

```python
from llm import create_client
from llm.config import DEFAULT_MODEL

# Create client
client = create_client(model_name=DEFAULT_MODEL)

# Normal mode (default) - Chat Completions API
result = client.complete("What is Python?")
print(result.get_answer_text())

# Specific mode - Token probability analysis
result = client.complete("What is the capital of China?", mode="specific")
result.print_analysis()  # Print token probability analysis
```

#### Using Emotion-Enhanced RAG

```python
from point_label.emotion import Emotion

# 提取情感向量
emotion = Emotion()
chunk = "I'm very happy!"
vector = emotion.extract(chunk)
print(f"Emotion Vector: {vector}")  # 30 维向量
```

### 使用记忆深度和温度

```python
from point_label.labels import Labels

# 提取记忆深度和温度
labels = Labels()
chunk = "I'm very happy!"
result = labels.extract(chunk, use_specific=True)

print(f"Emotion Vector: {result.emotion_vector}")
print(f"Memory Depth: {result.memory_depth}")  # 0~1，越大越深刻
print(f"Temperature: {result.temperature}")    # 0~1，越大波动越大
```

### 使用惊讶值

```python
from point_label.speed import Speed

# 计算惊讶值
speed = Speed()
chunk = "Quantum entanglement overturns our understanding of reality!"
surprise = speed.extract(chunk, aggregation="mean")
print(f"Surprise Value: {surprise}")  # 值越大越意外/重要
```

### 使用实体抽取

```python
from utils.extract_entitiy import Entity

# 提取实体和三元组
entity = Entity()
chunk = "Barack Obama was the 44th president of the United States."

# 提取实体
entities = entity.extract_entities(chunk)
print(f"Entities: {entities}")  # ['Barack Obama', 'United States']

# 提取三元组
triples = entity.extract_triples(chunk)
print(f"Triples: {triples}")  # [['Barack Obama', 'was', '44th president'], ...]

# 同时提取
result = entity.extract_all(chunk)
print(f"Entities: {result['entities']}")
print(f"Triples: {result['triples']}")
```

### 使用双曲空间存储和检索

```python
from poincare import HyperAmyStorage, HyperAmyRetrieval

# 创建存储
storage = HyperAmyStorage(db_path="./hyperamy_db")

# 存储点
point = Point(
    content="I'm very happy!",
    emotion_vector=emotion_vector,
    memory_depth=0.8,
    temperature=0.2
)
storage.add_point(point)

# 创建检索器
retrieval = HyperAmyRetrieval(storage)

# 检索
query = "happy"
results = retrieval.search(query, top_k=5)
for result in results:
    print(f"Content: {result.content}, Score: {result.score}")
```

### 使用情感增强的 HippoRAG

```python
from sentiment.hipporag_enhanced import HippoRAGEnhanced
from hipporag.utils.config_utils import BaseConfig
from llm.config import BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL

# Configure models
config = BaseConfig(
    save_dir="./outputs",
    llm_base_url=BASE_URL,
    llm_name=DEFAULT_MODEL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    embedding_base_url=BASE_URL,
)

# Create emotion-enhanced HippoRAG
hipporag = HippoRAGEnhanced(
    global_config=config,
    enable_emotion=True,
    emotion_weight=0.3,  # 30% emotion, 70% semantic
    emotion_model_name=DEFAULT_MODEL
)

# Index documents
docs = [
    "I'm thrilled about winning the competition! This is amazing!",
    "I'm devastated by the loss. Everything feels hopeless.",
    "The weather is nice today. It's a beautiful sunny day."
]
hipporag.index(docs=docs)

# Retrieve with emotion enhancement
queries = ["What makes people feel happy?", "What causes sadness?"]
results = hipporag.retrieve(queries=queries, num_to_retrieve=2)

# RAG QA with emotion awareness
qa_results, messages, metadata = hipporag.rag_qa(queries=queries)
```

## Project Structure

1. **测试运行方式**：始终在项目根目录下使用 `python -m test.xxx` 运行测试，不要修改 `sys.path` 或使用 `os.path`
2. **配置管理**：所有配置通过 `llm.config` 模块访问，不要直接读取环境变量
3. **模式选择**：默认使用 `normal` 模式（普通对话），需要 token 概率时使用 `mode="specific"`
4. **模型名称**：模型名称在代码中自定义，不作为环境变量，可以使用 `DEFAULT_MODEL` 和 `DEFAULT_EMBEDDING_MODEL` 作为默认值
5. **记忆深度计算**：`memory_depth = purity × normalized_magnitude`，其中纯度 = max(emotion_vector) / sum(emotion_vector)
6. **温度计算**：仅在 `use_specific=True` 时计算，需要 token 概率信息

## Core Modules

- `requests`：HTTP 请求
- `python-dotenv`：环境变量管理
- `numpy`：数值计算
- `pandas`：数据处理
- `chromadb`：向量数据库
- `hipporag`：检索增强生成框架（外部依赖）

## 版本历史

- **v1.2.0**：添加双曲空间存储与检索模块（poincare）
- **v1.1.0**：添加点标签模块（point_label）和实体抽取模块（utils）
- **v1.0.0**：初始版本，包含 LLM 客户端和情感分析模块
### `sentiment` Module

The emotion analysis module provides:

- **`emotion_vector.py`**: Extracts 28-dimensional emotion vectors from text using LLMs
- **`emotion_store.py`**: Manages persistent storage of emotion vectors using Parquet format
- **`hipporag_enhanced.py`**: `HippoRAGEnhanced` class that extends `HippoRAG` with emotion analysis

### `llm` Module

The LLM client module provides:

- **`completion_client.py`**: 
  - `CompletionClient`: Supports normal and specific modes
  - `create_client()`: Convenience function to create clients
  - `ChatResult`: Results for normal mode (Chat Completions API)
  - `CompletionResult`: Results for specific mode with token probabilities
- **`config.py`**: Unified API configuration management

### `hipporag` Module

The core RAG framework (based on [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)):

- **`HippoRAG.py`**: Main RAG framework class
- **`embedding_store.py`**: Embedding vector storage
- **`embedding_model/`**: Support for various embedding models (OpenAI, NV-Embed-v2, etc.)
- **`llm/`**: LLM inference classes (OpenAI GPT, Bedrock, Transformers, vLLM)
- **`evaluation/`**: Evaluation metrics for retrieval and QA

## Running Tests

**Important**: All tests should be run from the project root directory using `python -m`:

### Basic Tests

```bash
# Test token probability analysis (specific mode)
python -m test.test_infer

# Test Completion Client functionality
python -m test.test_completion_client
```

### Emotion Analysis Tests

```bash
# Test emotion vector extraction
python -m test.test_emotion

# Test BGE embedding and emotion description
python -m test.test_bge
```

### Integration Tests

```bash
# Test HippoRAG integration (small sample)
python -m test.test_integration

# Test dataset integration (real dataset)
python -m test.test_dataset_integration
```

## Usage Examples

### Example 1: Emotion Vector Extraction

```python
from sentiment.emotion_vector import EmotionExtractor

extractor = EmotionExtractor()
text = "I'm so happy and excited about this news!"
emotion_vector = extractor.extract_emotion_vector(text)
print(f"Emotion vector: {emotion_vector}")
```

### Example 2: Emotion-Enhanced Retrieval

```python
from sentiment.hipporag_enhanced import HippoRAGEnhanced
from hipporag.utils.config_utils import BaseConfig

# Initialize with emotion enhancement
config = BaseConfig(
    save_dir="./outputs",
    llm_base_url="https://llmapi.paratera.com/v1",
    llm_name="DeepSeek-V3.2",
    embedding_model_name="GLM-Embedding-2",
)

hipporag = HippoRAGEnhanced(
    global_config=config,
    enable_emotion=True,
    emotion_weight=0.3,  # Adjust emotion vs semantic weight
)

# Index documents
hipporag.index(docs=your_documents)

# Retrieve with emotion awareness
results = hipporag.retrieve(queries=your_queries)
```

### Example 3: Token Probability Analysis

```python
from llm import create_client

client = create_client(model_name="DeepSeek-V3.2")

# Get token-level probabilities
result = client.complete(
    "Explain quantum computing",
    mode="specific"
)

# Analyze token probabilities
result.print_analysis()
```

## Dependencies

### Required Dependencies

All required dependencies are listed in `requirements.txt`:

- `requests>=2.32.0`: HTTP requests
- `python-dotenv>=1.1.0`: Environment variable management
- `numpy>=1.26.0`: Numerical computing
- `pandas>=2.0.0`: Data processing
- `openai>=1.91.0`: OpenAI API client
- `httpx>=0.28.0`: Async HTTP client
- `pyarrow>=14.0.0` or `fastparquet>=2025.12.0`: Parquet file support
- `python-igraph>=0.11.0`: Graph processing
- `tenacity>=8.5.0`: Retry mechanism
- `tqdm>=4.66.0`: Progress bars

### Optional Dependencies

Install based on your use case:

- `transformers>=4.45.0`: Transformers model support
- `sentence-transformers>=2.2.0`: Sentence Transformers embedding
- `litellm>=1.73.0`: Bedrock support
- `torch>=2.0.0`: PyTorch support
- `vllm>=0.2.0`: VLLM offline inference
- `gritlm>=1.0.0`: GritLM embedding
- `outlines>=0.0.1`: Transformers offline mode

## Environment Alignment

To ensure consistent environments across collaborators:

1. **Use the same Python version**: Python 3.10.18
2. **Use the same dependency versions**: Run `pip install -r requirements.txt`
3. **Verify environment**: Run `python scripts/check_environment.py`

## Code Structure

The project follows a modular structure:

- **`sentiment/`**: Emotion analysis functionality
- **`llm/`**: LLM client and configuration
- **`hipporag/`**: Core RAG framework (based on HippoRAG)
- **`test/`**: Test suites for all modules
- **`scripts/`**: Utility scripts

## Key Differences from HippoRAG

HyperAmy extends HippoRAG with:

1. **Emotion Analysis**: 28-dimensional emotion vector extraction
2. **Emotion-Enhanced Retrieval**: Combines semantic and emotional similarity
3. **Emotion Storage**: Persistent storage of emotion vectors
4. **Token Probability**: Support for token-level probability analysis
5. **Enhanced API**: Improved error handling and robustness

## Notes

1. **Test Execution**: Always run tests from the project root using `python -m test.xxx`
2. **Configuration Management**: All configuration is accessed through `llm.config` module
3. **Mode Selection**: Default is `normal` mode (chat), use `mode="specific"` for token probabilities
4. **Model Names**: Model names are specified in code, not as environment variables

## Contributing

We welcome contributions! Please ensure:

1. Code follows Python 3.10+ standards
2. All tests pass
3. Environment alignment (use `requirements.txt`)
4. Documentation is updated

## Related Work

- [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG): The base RAG framework that HyperAmy extends
- [HippoRAG Paper](https://arxiv.org/abs/2405.14831): Original HippoRAG paper

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
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=hkujvAPVsg}
}
```

## License

MIT License - see LICENSE file for details

## Contact

Questions or issues? Please file an issue on [GitHub](https://github.com/sherkevin/HyperAmy/issues).

---

**HyperAmy**: Emotion-Enhanced RAG Framework built on HippoRAG
