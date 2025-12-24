# HyperAmy

一个集成了情感分析和 HippoRAG 的 LLM 应用框架，支持 token 级别的概率分析和情感增强的检索系统。

## 项目结构

```
HyperAmy/
├── llm/                    # LLM 客户端模块
│   ├── __init__.py         # 模块导出
│   ├── config.py           # 配置管理（从 .env 读取 API_KEY 和 BASE_URL）
│   ├── completion_client.py # LLM 客户端（支持 normal 和 specific 两种模式）
│   └── README.md           # LLM 模块详细文档
│
├── sentiment/              # 情感分析模块
│   ├── __init__.py
│   ├── emotion_vector.py   # 情感向量提取
│   ├── emotion_store.py    # 情感向量存储和管理
│   └── hipporag_enhanced.py # HippoRAG 增强版（集成情感分析）
│
├── test/                   # 测试文件
│   ├── test_infer.py       # 测试推理和 token 概率分析
│   ├── test_completion_client.py # 测试 Completion Client 功能
│   ├── test_emotion.py     # 测试情感向量提取
│   ├── test_bge.py         # 测试 BGE 嵌入和情感描述
│   ├── test_integration.py # 测试 HippoRAG 整合
│   └── test_dataset_integration.py # 测试数据集整合
│
└── hipporag/              # HippoRAG 框架（外部依赖）
```

## 环境要求

- **Python**: 3.10+ (推荐 3.10.18)
- **Conda**: 推荐使用 conda 管理环境

## 快速开始

### 1. 环境配置

#### 方式 1: 使用 Conda（推荐）

```bash
# 创建并激活 conda 环境
conda create -n Amygdala python=3.10.18
conda activate Amygdala

# 安装依赖
cd /path/to/hyperamy_source
pip install -r requirements.txt
```

#### 方式 2: 使用 pip

```bash
# 确保 Python 3.10+
python --version

# 安装依赖
pip install -r requirements.txt
```

#### 自动激活环境（可选）

项目已配置自动激活脚本，进入项目目录时会自动激活 Amygdala 环境：

```bash
# 首次设置（只需一次）
./setup_auto_activate.sh
source ~/.bashrc  # 或重新打开终端

# 之后每次进入项目目录会自动激活
cd /path/to/hyperamy_source
# 环境已自动激活！
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

#### 情感分析测试

```bash
# 测试情感向量提取
python -m test.test_emotion

# 测试 BGE 嵌入和情感描述
python -m test.test_bge
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

### sentiment 模块

- **`sentiment/emotion_vector.py`**：从文本中提取情感向量
- **`sentiment/emotion_store.py`**：情感向量的存储和管理
- **`sentiment/hipporag_enhanced.py`**：增强版 HippoRAG，集成情感分析功能

### test 模块

- **`test/test_infer.py`**：测试 LLM 推理和 token 概率分析（使用 specific 模式）
- **`test/test_completion_client.py`**：测试 Completion Client 的各种功能
- **`test/test_emotion.py`**：测试情感向量提取和分析
- **`test/test_bge.py`**：测试 BGE 嵌入和情感描述提取
- **`test/test_integration.py`**：测试 HippoRAG + 情感分析的整合功能
- **`test/test_dataset_integration.py`**：使用真实数据集测试整合功能

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
    enable_emotion=True,
    emotion_weight=0.3,
    emotion_model_name=llm_model_name
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

## 依赖说明

### 必需依赖

所有必需依赖已在 `requirements.txt` 中列出，包括：
- `requests`: HTTP 请求
- `python-dotenv`: 环境变量管理
- `numpy`: 数值计算
- `pandas`: 数据处理
- `openai`: OpenAI API 客户端
- `httpx`: 异步 HTTP 客户端
- `pyarrow` 或 `fastparquet`: Parquet 文件支持
- `python-igraph`: 图处理
- `tenacity`: 重试机制
- `tqdm`: 进度条

### 可选依赖

根据使用场景可选安装：
- `transformers`: Transformers 模型支持
- `sentence-transformers`: Sentence Transformers embedding
- `litellm`: Bedrock 支持
- `torch`: PyTorch 支持
- `vllm`: VLLM 离线推理
- `gritlm`: GritLM embedding
- `outlines`: Transformers 离线模式

### 环境对齐

为确保协作者环境一致，请：

1. **使用相同的 Python 版本**: Python 3.10.18
2. **使用相同的依赖版本**: 运行 `pip install -r requirements.txt`
3. **验证环境**: 运行 `python scripts/check_environment.py`

### 环境检查

运行环境检查脚本验证配置：
```bash
python scripts/check_environment.py
```

应该看到：
- ✅ Python 版本: 3.10.18
- ✅ CONDA_DEFAULT_ENV: Amygdala (如果使用 conda)
- ✅ 所有必需依赖已安装
- ✅ API 配置正确
