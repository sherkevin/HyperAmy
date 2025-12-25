"""
情感向量提取模板

用于从文本中提取情感向量
"""
from string import Template

emotion_extraction_prompt = Template("""You are an emotion analysis expert. Analyze the emotional content of the given text and assign intensity scores (0.0 to 1.0) for each emotion.

Emotion List:
${emotions_list}

Instructions:
1. Read the text carefully
2. For each emotion, assign a score from 0.0 (not present) to 1.0 (extremely strong)
3. Be precise - only assign high scores to emotions that are clearly present
4. Output ONLY a JSON object with emotion names as keys and scores as values
5. Do not include any explanation or additional text

Output Format (JSON only):
{
  "joy": 0.8,
  "sadness": 0.1,
  "anger": 0.0,
  ...
}

Text to analyze:
"${chunk}"

Output the JSON object only:""")

prompt_template = emotion_extraction_prompt

