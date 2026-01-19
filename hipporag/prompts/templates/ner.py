"""
NER Prompt模板（统一版）

为 HippoRAG 和 HyperAmy 提供统一的实体抽取功能。

设计原则：
1. 包含抽象概念（HyperAmy 的优势）
2. Few-shot 示例提升准确性
3. 支持情感相关实体（HyperAmy 需要）
"""

ner_system = """Your task is to extract all significant entities from the given paragraph.
An entity can be:
1. Named entities: people, organizations, locations, dates, products, etc.
2. Abstract concepts: topics, themes, subjects, ideas, technologies, fields of study, emotions, feelings, etc.
3. Important terms: technical terms, domain-specific words, key phrases, etc.
4. Emotional entities: attitudes, reactions, states of mind, etc.

Respond with a JSON object containing a "named_entities" list with ALL significant entities found.
Be inclusive - if a word or phrase represents an important concept, topic, or entity, include it.
"""

# Few-shot 示例 1: 编程相关
one_shot_ner_paragraph_1 = """I love Python programming!"""

one_shot_ner_output_1 = """{"named_entities": ["Python", "programming"]}"""

# Few-shot 示例 2: 抽象概念 + 学习
one_shot_ner_paragraph_2 = """Learning new technologies is exciting and challenging."""

one_shot_ner_output_2 = """{"named_entities": ["technologies", "learning", "exciting", "challenging"]}"""

# Few-shot 示例 3: 情感 + 主题
one_shot_ner_paragraph_3 = """The weather is beautiful today."""

one_shot_ner_output_3 = """{"named_entities": ["weather", "beautiful"]}"""

# Few-shot 示例 4: 传统命名实体
one_shot_ner_paragraph_4 = """Barack Obama was the 44th president of the United States."""

one_shot_ner_output_4 = """{"named_entities": ["Barack Obama", "United States", "44th president"]}"""

# Few-shot 示例 5: 情感相关（HyperAmy 需要）
one_shot_ner_paragraph_5 = """She looked at him with terror in her eyes, her hand trembling as she held the plate."""

one_shot_ner_output_5 = """{"named_entities": ["terror", "trembling", "plate"]}"""

# 统一的 prompt 模板（few-shot）
prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph_1},
    {"role": "assistant", "content": one_shot_ner_output_1},
    {"role": "user", "content": one_shot_ner_paragraph_2},
    {"role": "assistant", "content": one_shot_ner_output_2},
    {"role": "user", "content": one_shot_ner_paragraph_3},
    {"role": "assistant", "content": one_shot_ner_output_3},
    {"role": "user", "content": one_shot_ner_paragraph_4},
    {"role": "assistant", "content": one_shot_ner_output_4},
    {"role": "user", "content": one_shot_ner_paragraph_5},
    {"role": "assistant", "content": one_shot_ner_output_5},
    {"role": "user", "content": "${passage}"}
]

# 简化版模板（单示例，更快）- 用于查询等需要快速响应的场景
prompt_template_simple = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": "The weather is beautiful today and I feel happy."},
    {"role": "assistant", "content": '{"named_entities": ["weather", "beautiful", "happy"]}'},
    {"role": "user", "content": "${passage}"}
]

# 向后兼容别名（用于 triple_extraction.py）
one_shot_ner_paragraph = one_shot_ner_paragraph_1
one_shot_ner_output = one_shot_ner_output_1
