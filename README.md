# HyperAmy

## 快速开始

### 1. 环境配置

确保已安装必要的依赖：
```bash
pip install requests python-dotenv numpy
```

### 2. 配置环境变量

在 `llm/.env` 文件中配置 API 密钥：
```bash
API_KEY=your_api_key_here
API_URL_COMPLETIONS=https://llmapi.paratera.com/v1/completions
API_URL_CHAT=https://llmapi.paratera.com/v1/chat/completions
API_URL_EMBEDDINGS=https://llmapi.paratera.com/v1/embeddings
DEFAULT_MODEL=DeepSeek-V3.2
```

### 3. 运行测试

项目提供了多个测试脚本，可以按需运行：

#### 基础测试
```bash
# 测试 Completion Client（推理和概率分析）
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
# 测试 HippoRAG 整合
python -m test.test_integration

# 测试数据集整合
python -m test.test_dataset_integration
```

## 测试说明

- `test_infer.py`: 测试 LLM 推理和 token 概率分析
- `test_completion_client.py`: 测试 Completion API 客户端功能
- `test_emotion.py`: 测试情感向量提取和分析
- `test_bge.py`: 测试 BGE 嵌入和情感描述提取
