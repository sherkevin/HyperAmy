"""
改进的NER Prompt模板

支持更广泛的实体类型，包括抽象概念
"""

ner_system_improved = """Your task is to extract all significant entities from the given paragraph.
An entity can be:
1. Named entities: people, organizations, locations, dates, products, etc.
2. Abstract concepts: topics, themes, subjects, ideas, technologies, fields of study, etc.
3. Important terms: technical terms, domain-specific words, key phrases, etc.

Respond with a JSON object containing a "named_entities" list with all entities found.
"""

# 示例1: 包含具体实体
one_shot_ner_example_1 = """I love Python programming!"""

one_shot_output_1 = """{"named_entities": ["Python", "programming"]}"""

# 示例2: 包含抽象概念
one_shot_ner_example_2 = """Learning new technologies is exciting and challenging."""

one_shot_output_2 = """{"named_entities": ["technologies", "learning"]}"""

# 示例3: 包含情感和主题
one_shot_ner_example_3 = """The weather is beautiful today."""

one_shot_output_3 = """{"named_entities": ["weather"]}"""

# 示例4: 传统命名实体
one_shot_ner_example_4 = """Barack Obama was the 44th president of the United States."""

one_shot_output_4 = """{"named_entities": ["Barack Obama", "United States", "44th president"]}"""

# 改进的prompt模板（使用few-shot）
prompt_template_improved = [
    {"role": "system", "content": ner_system_improved},
    {"role": "user", "content": one_shot_ner_example_1},
    {"role": "assistant", "content": one_shot_output_1},
    {"role": "user", "content": one_shot_ner_example_2},
    {"role": "assistant", "content": one_shot_output_2},
    {"role": "user", "content": one_shot_ner_example_3},
    {"role": "assistant", "content": one_shot_output_3},
    {"role": "user", "content": one_shot_ner_example_4},
    {"role": "assistant", "content": one_shot_output_4},
    {"role": "user", "content": "${passage}"}
]

# 简化版（保持原有结构，只改进system prompt和one-shot示例）
prompt_template_simple_improved = [
    {"role": "system", "content": ner_system_improved},
    {"role": "user", "content": "The weather is beautiful today and I feel happy."},
    {"role": "assistant", "content": '{"named_entities": ["weather", "happy"]}'},
    {"role": "user", "content": "${passage}"}
]
