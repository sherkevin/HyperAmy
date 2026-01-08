ner_system = """Your task is to extract all significant entities from the given paragraph.
An entity can be:
1. Named entities: people, organizations, locations, dates, products, etc.
2. Abstract concepts: topics, themes, subjects, ideas, technologies, fields of study, emotions, feelings, etc.
3. Important terms: technical terms, domain-specific words, key phrases, etc.

Respond with a JSON object containing a "named_entities" list with ALL significant entities found.
Be inclusive - if a word or phrase represents an important concept, topic, or entity, include it.
"""

# 改进的示例 - 包含抽象概念
one_shot_ner_paragraph = """The weather is beautiful today, and I feel happy about learning new programming technologies."""

one_shot_ner_output = """{"named_entities":
    ["weather", "beautiful", "happy", "learning", "programming", "technologies"]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]
