# 实体粒度数据集生成脚本改进版 v2

## 改进内容

### 1. 过滤章节标题 ✅

**问题**：原始数据中很多文本是章节标题（如"Marseilles—The Arrival\nChapter 2"），导致提取实体时"Chapter"被频繁提取。

**解决方案**：
- 添加 `is_chapter_title()` 函数检测章节标题
- 添加 `clean_text()` 函数清理文本，移除章节标题
- 在 `process_single_text()` 开始时调用 `clean_text()` 过滤无用文本

**代码位置**：
```python
def is_chapter_title(text: str) -> bool:
    """判断是否是章节标题（包含Chapter的短文本）"""
    text_clean = text.strip()
    if len(text_clean) < 50 and re.search(r'\bChapter\s+\d+', text_clean, re.IGNORECASE):
        return True
    # ... 更多检测逻辑

def clean_text(text: str) -> Optional[str]:
    """清理文本，移除章节标题等无用内容"""
    # 过滤章节标题
    if is_chapter_title(text):
        return None
    # ... 清理逻辑
```

### 2. soft_label不进行全局归一化 ✅

**问题**：原始代码对28维向量进行全局归一化（sum=1.0），这导致模长固定，无法反映情绪强度变化。

**改进**：
- **不进行全局归一化**：每个维度独立限制在[0, 1]范围
- **使用L2-norm计算intensity**：更合理反映情绪强度
- **修改prompt**：明确告诉LLM每个分数是独立的，不需要归一化

**代码变化**：
```python
# 旧代码（错误）：
# 归一化确保和为1.0
total = sum(soft_label)
if total > 0:
    soft_label = [s / total for s in soft_label]

# 新代码（正确）：
# 限制每个维度在[0, 1]范围，但不做全局归一化
soft_label = [max(0.0, min(1.0, float(s))) for s in soft_label]

# 计算intensity (L2-norm，更合理)
intensity = float(np.linalg.norm(soft_label))
```

**Prompt改进**：
```
Provide emotion scores as a JSON array of 28 numbers, where each number is between 0.0 and 1.0, representing the intensity of that emotion. Each score should be independent (no normalization required).
```

### 3. 支持QA对数据 ✅

**需求**：将QA对加入到训练集，确保Q和A的实体情绪向量尽可能相近。

**实现**：
- 添加 `EMOTION_PROMPT_QA_TEMPLATE`：专门用于QA对的prompt
- 在prompt中提示LLM考虑Q和A的情绪相似性（但不要太明显）
- `extract_emotion_soft_label()` 支持 `is_qa_pair` 和 `related_text` 参数
- `generate_entity_granularity_dataset()` 支持加载QA文件并处理

**QA Prompt特点**：
```
Analyze the emotional content of the following text span, which is part of a question-answer pair. Consider that questions and their corresponding answers often share similar emotional undertones.

Related text (from the same QA pair): {related_text}

When the text span and related text share emotional context, reflect that in the scores.
```

**使用方式**：
```bash
python scripts/generate_entity_granularity_dataset_v2.py \
  --training-files data/training/monte_cristo_train_full.jsonl \
  --qa-files data/public_benchmark/monte_cristo_qa_full.json \
  --qa-ratio 0.3 \
  --output data/training/entity_granularity/entity_granularity_v2.jsonl
```

### 4. 支持多数据源 ✅

**需求**：支持从多个数据源加载文本数据。

**实现**：
- `--training-files` 参数支持多个文件（使用 `nargs='+'`）
- `--qa-files` 参数支持多个QA文件
- `generate_entity_granularity_dataset()` 函数处理多个数据源
- 自动合并所有数据源

**使用方式**：
```bash
python scripts/generate_entity_granularity_dataset_v2.py \
  --training-files \
    data/training/monte_cristo_train_full.jsonl \
    data/training/other_texts.jsonl \
  --qa-files \
    data/public_benchmark/monte_cristo_qa_full.json \
    data/benchmarks/instinct_qa.json \
  --output data/training/entity_granularity/entity_granularity_v2.jsonl
```

## 主要API变化

### 函数签名变化

**旧版本**：
```python
def generate_entity_granularity_dataset(
    input_file: Path,
    output_file: Path,
    ...
)
```

**新版本**：
```python
def generate_entity_granularity_dataset(
    training_files: List[Path],
    qa_files: List[Path],
    output_file: Path,
    qa_ratio: float = 0.3,
    ...
)
```

### 命令行参数变化

- `--input` → `--training-files` (支持多个文件)
- 新增 `--qa-files` (支持多个文件)
- 新增 `--qa-ratio` (控制QA对比例，默认0.3)

## 数据格式变化

### soft_label格式

**旧格式**（归一化后）：
```json
{
  "soft_label": [0.01, 0.02, 0.85, 0.05, ...],  // sum = 1.0
  "intensity": 0.85  // Max-Norm
}
```

**新格式**（不归一化）：
```json
{
  "soft_label": [0.01, 0.02, 0.85, 0.05, ...],  // 每个维度在[0,1]，sum不一定=1.0
  "intensity": 0.92  // L2-norm
}
```

## 使用示例

### 基础使用
```bash
python scripts/generate_entity_granularity_dataset_v2.py \
  --training-files data/training/monte_cristo_train_full.jsonl \
  --output data/training/entity_granularity/entity_granularity_v2.jsonl
```

### 包含QA对
```bash
python scripts/generate_entity_granularity_dataset_v2.py \
  --training-files data/training/monte_cristo_train_full.jsonl \
  --qa-files data/public_benchmark/monte_cristo_qa_full.json \
  --qa-ratio 0.3 \
  --output data/training/entity_granularity/entity_granularity_v2.jsonl
```

### 多数据源
```bash
python scripts/generate_entity_granularity_dataset_v2.py \
  --training-files \
    data/training/monte_cristo_train_full.jsonl \
    data/training/other_source.jsonl \
  --qa-files \
    data/public_benchmark/monte_cristo_qa_full.json \
    data/benchmarks/instinct_qa.json \
  --max-samples-per-source 1000 \
  --qa-ratio 0.3 \
  --output data/training/entity_granularity/entity_granularity_v2.jsonl
```

## 预期效果

1. **数据质量提升**：
   - 过滤掉章节标题等无用文本
   - 文本质量更高，情绪更丰富

2. **情绪强度更准确**：
   - soft_label的模长可以变化，更好地反映情绪强度
   - intensity使用L2-norm，更合理

3. **QA检索性能提升**：
   - QA对的实体情绪向量更相近
   - 训练后的emotion embedding model在QA检索时性能更好

4. **数据源更丰富**：
   - 支持多个数据源
   - 可以混合不同类型的数据（训练文本+QA对）
