"""
情感视角描述生成模板

用于为实体生成情感视角描述
"""
from string import Template

affective_description_prompt = Template("""Extract emotional keywords about "${entity}" from the sentence below.

REQUIREMENTS:
- Output ONLY a comma-separated list of emotion words (3-8 words)
- Use precise emotional vocabulary (e.g., joy, sadness, anger, trust, admiration, fear, hope, pride)
- Focus on core emotional essence, ignore factual details
- Each word should be a distinct emotion or affective state
- Output format: word1, word2, word3, ...

Sentence: "${sentence}"
Entity: "${entity}"

Emotional keywords:""")

prompt_template = affective_description_prompt

