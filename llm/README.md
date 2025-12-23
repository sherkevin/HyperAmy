# LLM Completion Client

封装了使用 Completion 接口调用大语言模型的功能，支持获取 token 级别的概率信息。

## 功能特性

- ✅ 支持 Completion API 调用
- ✅ 获取 token 级别的 logprob 和 probability
- ✅ 分离 prompt 和 answer 的 token 信息
- ✅ 支持自定义 prompt 模板
- ✅ 支持自定义参数（temperature, max_tokens 等）
- ✅ 提供便捷的快速回答方法

## 快速开始

### 基本使用

```python
from llm import create_client

# 创建客户端
client = create_client(
    api_key="your-api-key",
    model_name="DeepSeek-V3.2"
)

# 获取完整结果（包含 token 概率信息）
result = client.complete("中国的首都是哪里？")

# 获取回答文本
print(result.get_answer_text())

# 打印详细分析（token 概率）
result.print_analysis()
```

### 快速获取回答

如果只需要回答文本，不需要 token 概率信息：

```python
answer = client.get_answer("Python 是什么？")
print(answer)
```

### 自定义 Prompt 模板

```python
custom_template = "问题：{query}\n回答："
result = client.complete(
    "什么是量子力学？",
    prompt_template=custom_template
)
```

### 自定义参数

```python
result = client.complete(
    "解释一下机器学习",
    max_tokens=200,
    temperature=0.5,
    stop=["\n\n", "User:"]
)
```

## API 参考

### CompletionClient

#### 初始化参数

- `api_key` (str): API 密钥
- `api_url` (str): API 地址，默认为 `"https://llmapi.paratera.com/v1/completions"`
- `model_name` (str): 模型名称，默认为 `"DeepSeek-V3.2"`
- `default_max_tokens` (int): 默认最大 token 数，默认为 100
- `default_temperature` (float): 默认温度参数，默认为 0.7
- `default_stop` (List[str]): 默认停止词列表

#### 方法

##### `complete(query, **kwargs) -> CompletionResult`

调用 Completion API，返回完整结果。

**参数：**
- `query` (str): 用户查询
- `prompt_template` (str, optional): Prompt 模板
- `max_tokens` (int, optional): 最大 token 数
- `temperature` (float, optional): 温度参数
- `stop` (List[str], optional): 停止词列表
- `logprobs` (int): Log 概率数量，默认为 1
- `echo` (bool): 是否回显 prompt，默认为 True
- `**kwargs`: 其他 API 参数

**返回：** `CompletionResult` 对象

##### `get_answer(query, **kwargs) -> str`

快速获取回答文本（不返回详细概率信息）。

**参数：** 同 `complete` 方法

**返回：** 回答文本字符串

### CompletionResult

#### 属性

- `prompt_tokens` (List[TokenInfo]): Prompt 部分的 token 信息
- `answer_tokens` (List[TokenInfo]): Answer 部分的 token 信息
- `answer_text` (str): 完整的回答文本
- `usage` (Dict[str, int]): Token 使用统计
- `raw_response` (Dict[str, Any]): 原始 API 响应

#### 方法

- `get_prompt_text() -> str`: 获取完整的 prompt 文本
- `get_answer_text() -> str`: 获取完整的回答文本
- `print_analysis()`: 打印详细的分析结果（token 概率）

### TokenInfo

#### 属性

- `token` (str): Token 文本
- `logprob` (Optional[float]): Log 概率
- `probability` (Optional[float]): 概率值（exp(logprob)）

## 示例

完整示例请参考 `test/test_completion_client.py` 和 `test/test_infer.py`。

