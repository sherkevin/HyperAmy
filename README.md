# HyperAmy

一个集成了情感分析、实体抽取和双曲空间存储的 LLM 应用框架，支持 token 级别的概率分析、情感增强的检索系统和基于 Poincaré 球的双曲空间记忆存储。

## 项目结构

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

## 快速开始

### 1. 环境配置

确保已安装必要的依赖：
```bash
uv sync
```

### 2. 配置环境变量

在 `llm/.env` 文件中配置 API 密钥和基础 URL：
```bash
API_KEY=your_api_key_here
BASE_URL=https://llmapi.paratera.com/v1
```

**注意**：
- `.env` 文件只需要包含 `API_KEY` 和 `BASE_URL`
- 模型名称（model name）在代码中自定义，不作为环境变量
- 配置通过 `llm.config` 模块统一管理

### 3. LLM 使用方式

#### 使用 config 模块

所有配置都通过 `llm.config` 模块访问：

```python
from llm.config import API_KEY, BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL
from llm import create_client

# 创建客户端（默认使用 normal 模式）
client = create_client(model_name=DEFAULT_MODEL)

# 普通对话模式（normal）- 默认模式
result = client.complete("你好")
print(result.get_answer_text())

# 获取 token 概率模式（specific）
result = client.complete("中国的首都是哪里？", mode="specific")
result.print_analysis()  # 打印 token 概率分析
```

#### 两种模式说明

- **normal 模式**（默认）：使用 Chat Completions API，返回 `ChatResult`，适合普通对话
- **specific 模式**：使用 Completions API，返回 `CompletionResult`，包含 `prompt_tokens`、`answer_tokens` 和 `print_analysis()` 方法，用于获取 token 级别的概率信息

### 4. 运行测试

**重要**：所有测试都在项目根目录下使用 `python -m` 方式运行，不要使用 `os.path` 修改路径。

#### 基础测试

```bash
# 测试推理和 token 概率分析（使用 specific 模式）
python -m test.test_infer

# 测试 Completion Client 的完整功能
python -m test.test_completion_client
```

#### 点标签模块测试

```bash
# 测试情感向量提取
python -m test.test_emotion

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

```python
from llm import create_client
from llm.config import DEFAULT_MODEL

# 创建客户端
client = create_client(model_name=DEFAULT_MODEL)

# 普通对话（normal 模式）
result = client.complete("Python 是什么？")
print(result.get_answer_text())

# 获取 token 概率（specific 模式）
result = client.complete("中国的首都是哪里？", mode="specific")
result.print_analysis()
```

### 使用情感分析

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

# 配置模型（在代码中自定义）
llm_model_name = DEFAULT_MODEL
embedding_model_name = DEFAULT_EMBEDDING_MODEL

config = BaseConfig(
    save_dir="./outputs",
    llm_base_url=BASE_URL,
    llm_name=llm_model_name,
    embedding_model_name=embedding_model_name,
    embedding_base_url=BASE_URL,
)

# 创建增强版 HippoRAG
hipporag = HippoRAGEnhanced(
    global_config=config,
    enable_sentiment=True,
    sentiment_weight=0.3,
    sentiment_model_name=llm_model_name
)

# 索引文档
hipporag.index(docs=["文档1", "文档2"])

# 检索
results = hipporag.retrieve(queries=["查询1"])
```

## 注意事项

1. **测试运行方式**：始终在项目根目录下使用 `python -m test.xxx` 运行测试，不要修改 `sys.path` 或使用 `os.path`
2. **配置管理**：所有配置通过 `llm.config` 模块访问，不要直接读取环境变量
3. **模式选择**：默认使用 `normal` 模式（普通对话），需要 token 概率时使用 `mode="specific"`
4. **模型名称**：模型名称在代码中自定义，不作为环境变量，可以使用 `DEFAULT_MODEL` 和 `DEFAULT_EMBEDDING_MODEL` 作为默认值
5. **记忆深度计算**：`memory_depth = purity × normalized_magnitude`，其中纯度 = max(emotion_vector) / sum(emotion_vector)
6. **温度计算**：仅在 `use_specific=True` 时计算，需要 token 概率信息

## 依赖

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
