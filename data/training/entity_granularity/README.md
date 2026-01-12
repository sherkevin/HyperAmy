# 实体粒度情绪数据集

## 数据集格式

每个样本是一个JSON对象，格式如下：

```json
{
  "text": "The cat ate the cat food.",
  "targets": [
    {
      "span_text": "cat",
      "char_start": 4,
      "char_end": 7,
      "soft_label": [0.01, 0.02, 0.85, 0.05, ...],  // 28维概率向量
      "intensity": 0.85  // Max-Norm，即soft_label的最大值
    },
    {
      "span_text": "cat food",
      "char_start": 16,
      "char_end": 24,
      "soft_label": [0.01, 0.02, 0.05, 0.85, ...],  // 28维概率向量
      "intensity": 0.85
    }
  ]
}
```

## 字段说明

- `text`: 完整文本内容
- `targets`: 实体列表，每个实体包含：
  - `span_text`: 实体文本span
  - `char_start`: 实体在文本中的开始字符位置（从0开始）
  - `char_end`: 实体在文本中的结束字符位置（不包含）
  - `soft_label`: 28维情绪概率向量（和为1.0）
  - `intensity`: 情绪强度（Max-Norm，即soft_label的最大值）

## 28维情绪列表

1. admiration (钦佩)
2. amusement (愉悦)
3. approval (赞同)
4. caring (关心)
5. desire (渴望)
6. excitement (兴奋)
7. gratitude (感激)
8. joy (快乐)
9. love (爱)
10. optimism (乐观)
11. pride (骄傲)
12. relief (宽慰)
13. anger (愤怒)
14. annoyance (烦恼)
15. disappointment (失望)
16. disapproval (不赞同)
17. disgust (厌恶)
18. embarrassment (尴尬)
19. fear (恐惧)
20. grief (悲伤)
21. nervousness (紧张)
22. remorse (悔恨)
23. sadness (悲伤)
24. confusion (困惑)
25. curiosity (好奇)
26. realization (领悟)
27. surprise (惊讶)
28. neutral (中性)

## 数据生成

使用 `scripts/generate_entity_granularity_dataset.py` 生成：

```bash
python scripts/generate_entity_granularity_dataset.py \
  --input data/training/monte_cristo_train_full.jsonl \
  --output data/training/entity_granularity/entity_granularity_monte_cristo.jsonl \
  --max-samples 1000 \
  --max-entities 10 \
  --max-workers 10
```

## 数据验证

使用 `test/test_entity_granularity_dataset.py` 验证：

```bash
python test/test_entity_granularity_dataset.py \
  --dataset data/training/entity_granularity/entity_granularity_monte_cristo.jsonl
```

## 注意事项

1. 实体提取使用spaCy NER，如果spaCy不可用，会回退到基于规则的提取
2. 情绪标签通过LLM API提取，需要有效的API key
3. 数据生成支持断点续传（通过进度文件）
4. 情绪向量提取结果会被缓存，避免重复API调用

