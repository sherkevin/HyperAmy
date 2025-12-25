# Prompts 模块

统一管理项目中的 prompt 模板，参照 HippoRAG 的组织形式。

## 目录结构

```
prompts/
├── __init__.py                    # 模块导出
├── prompt_template_manager.py     # Prompt 模板管理器
├── templates/                     # Prompt 模板目录
│   ├── __init__.py
│   ├── affective_description.py  # 情感视角描述生成模板
│   └── emotion_extraction.py     # 情感向量提取模板
└── README.md                      # 本文档
```

## 使用方法

### 1. 创建模板文件

在 `templates/` 目录下创建 Python 文件，定义 `prompt_template` 变量：

```python
# prompts/templates/my_template.py
from string import Template

my_prompt = Template("""Your task is to ${task}.

Input: ${input}

Output:""")

prompt_template = my_prompt
```

### 2. 使用模板管理器

```python
from prompts import PromptTemplateManager

# 初始化管理器（自动加载 templates 目录下的所有模板）
manager = PromptTemplateManager()

# 列出所有可用模板
print(manager.list_template_names())  # ['affective_description', 'emotion_extraction']

# 渲染模板
prompt = manager.render(
    name='affective_description',
    sentence="I love Apple products.",
    entity="Apple"
)
```

### 3. 在类中使用

```python
from prompts import PromptTemplateManager

class MyClass:
    def __init__(self):
        self.prompt_template_manager = PromptTemplateManager()
    
    def my_method(self, input_text):
        prompt = self.prompt_template_manager.render(
            name='my_template',
            task="analyze",
            input=input_text
        )
        # 使用 prompt...
```

## 模板类型

### 字符串模板（String Template）

使用 `Template` 类，支持 `${variable}` 占位符：

```python
from string import Template

prompt_template = Template("Hello ${name}, your task is ${task}.")
```

### 聊天历史模板（Chat History Template）

使用列表格式，支持多轮对话：

```python
prompt_template = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "${query}"}
]
```

## 现有模板

### 1. `affective_description`

为实体生成情感视角描述。

**参数**：
- `sentence`: 原始句子
- `entity`: 实体名称

**使用示例**：
```python
prompt = manager.render(
    name='affective_description',
    sentence="Barack Obama was the 44th president.",
    entity="Barack Obama"
)
```

### 2. `emotion_extraction`

从文本中提取情感向量。

**参数**：
- `emotions_list`: 情绪列表（逗号分隔）
- `chunk`: 输入文本片段

**使用示例**：
```python
prompt = manager.render(
    name='emotion_extraction',
    emotions_list="joy, sadness, anger, ...",
    chunk="I'm very happy!"
)
```

## 注意事项

1. **模板文件命名**：模板文件名（不含 `.py`）即为模板名称
2. **必须定义 `prompt_template`**：每个模板文件必须定义 `prompt_template` 变量
3. **自动加载**：`PromptTemplateManager` 初始化时自动加载 `templates/` 目录下的所有 `.py` 文件
4. **占位符格式**：使用 `${variable_name}` 格式定义占位符

