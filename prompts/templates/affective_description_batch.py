"""
批量情感视角描述生成模板

方案六A：一次性为所有实体生成情感描述

优化：将 N 次 LLM 调用减少到 1 次
"""
from string import Template

affective_description_batch_prompt = Template("""Extract emotional keywords for each entity from the sentence below.

REQUIREMENTS:
- For EACH entity, output a comma-separated list of emotion words (3-8 words per entity)
- Use precise emotional vocabulary (e.g., joy, sadness, anger, trust, admiration, fear, hope, pride)
- Focus on core emotional essence, ignore factual details
- Each word should be a distinct emotion or affective state
- Output format: Entity: word1, word2, word3, ...

Sentence: "${sentence}"

Entities:
${entities_list}

Output each entity with its emotional keywords:""")

prompt_template = affective_description_batch_prompt
