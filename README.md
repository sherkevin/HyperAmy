# HyperAmy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HyperAmy is an emotion-enhanced RAG framework built on top of [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), integrating emotion analysis capabilities to enable LLMs to understand and leverage emotional context in retrieval-augmented generation tasks.

## Overview

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
# Create and activate conda environment
conda create -n Amygdala python=3.10.18
conda activate Amygdala

# Install dependencies
cd /path/to/hyperamy_source
pip install -r requirements.txt
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

You should see:
- ✅ Python version: 3.10.18
- ✅ All required dependencies installed
- ✅ API configuration correct

## Quick Start

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

```
HyperAmy/
├── llm/                          # LLM client module
│   ├── __init__.py
│   ├── config.py                 # Configuration management (reads from .env)
│   ├── completion_client.py      # LLM client (normal and specific modes)
│   └── README.md                 # LLM module documentation
│
├── sentiment/                    # Emotion analysis module
│   ├── __init__.py
│   ├── emotion_vector.py         # Emotion vector extraction
│   ├── emotion_store.py          # Emotion vector storage and management
│   └── hipporag_enhanced.py     # Enhanced HippoRAG with emotion analysis
│
├── test/                         # Test files
│   ├── test_infer.py            # Token probability analysis tests
│   ├── test_completion_client.py # Completion client tests
│   ├── test_emotion.py          # Emotion vector extraction tests
│   ├── test_bge.py              # BGE embedding and emotion description tests
│   ├── test_integration.py      # HippoRAG integration tests
│   └── test_dataset_integration.py # Dataset integration tests
│
├── hipporag/                     # HippoRAG framework (external dependency)
│   ├── HippoRAG.py              # Main RAG framework
│   ├── embedding_store.py       # Embedding storage
│   ├── embedding_model/          # Embedding model implementations
│   ├── llm/                     # LLM inference classes
│   ├── evaluation/              # Evaluation metrics
│   └── utils/                   # Utility functions
│
├── scripts/                      # Utility scripts
│   └── check_environment.py     # Environment verification script
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Core Modules

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
