"""
统一实体抽取 Prompt 模板

为 HippoRAG 和 HyperAmy 提供统一的实体抽取功能。

设计原则：
1. 包含抽象概念（HyperAmy 的优势）
2. Few-shot 示例提升准确性
3. 与 HippoRAG 格式兼容
4. 支持情感相关实体（HyperAmy 需要）
"""

ner_system_unified = """Your task is to extract all significant entities from the given paragraph.
An entity can be:
1. Named entities: people, organizations, locations, dates, products, etc.
2. Abstract concepts: topics, themes, subjects, ideas, technologies, fields of study, emotions, feelings, etc.
3. Important terms: technical terms, domain-specific words, key phrases, etc.
4. Emotional entities: attitudes, reactions, states of mind, etc.

Respond with a JSON object containing a "named_entities" list with ALL significant entities found.
Be inclusive - if a word or phrase represents an important concept, topic, or entity, include it.
"""

# Few-shot 示例 1: 编程相关
one_shot_ner_example_1 = """I love Python programming!"""

one_shot_output_1 = """{"named_entities": ["Python", "programming"]}"""

# Few-shot 示例 2: 抽象概念 + 学习
one_shot_ner_example_2 = """Learning new technologies is exciting and challenging."""

one_shot_output_2 = """{"named_entities": ["technologies", "learning", "exciting", "challenging"]}"""

# Few-shot 示例 3: 情感 + 主题
one_shot_ner_example_3 = """The weather is beautiful today."""

one_shot_output_3 = """{"named_entities": ["weather", "beautiful"]}"""

# Few-shot 示例 4: 传统命名实体
one_shot_ner_example_4 = """Barack Obama was the 44th president of the United States."""

one_shot_output_4 = """{"named_entities": ["Barack Obama", "United States", "44th president"]}"""

# Few-shot 示例 5: 情感相关（HyperAmy 需要）
one_shot_ner_example_5 = """She looked at him with terror in her eyes, her hand trembling as she held the plate."""

one_shot_output_5 = """{"named_entities": ["terror", "trembling", "plate"]}"""

# 统一的 prompt 模板（few-shot）
prompt_template_unified = [
    {"role": "system", "content": ner_system_unified},
    {"role": "user", "content": one_shot_ner_example_1},
    {"role": "assistant", "content": one_shot_output_1},
    {"role": "user", "content": one_shot_ner_example_2},
    {"role": "assistant", "content": one_shot_output_2},
    {"role": "user", "content": one_shot_ner_example_3},
    {"role": "assistant", "content": one_shot_output_3},
    {"role": "user", "content": one_shot_ner_example_4},
    {"role": "assistant", "content": one_shot_output_4},
    {"role": "user", "content": one_shot_ner_example_5},
    {"role": "assistant", "content": one_shot_output_5},
    {"role": "user", "content": "${passage}"}
]

# 简化版模板（单示例，更快）
prompt_template_unified_simple = [
    {"role": "system", "content": ner_system_unified},
    {"role": "user", "content": "The weather is beautiful today and I feel happy."},
    {"role": "assistant", "content": '{"named_entities": ["weather", "beautiful", "happy"]}'},
    {"role": "user", "content": "${passage}"}
]

__all__ = [
    'ner_system_unified',
    'prompt_template_unified',
    'prompt_template_unified_simple'
]
