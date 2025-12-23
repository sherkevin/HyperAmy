"""
情感向量提取模块

从 HyperAmy 提取的核心功能，用于提取文本的情感向量。
"""
import requests
import numpy as np
import json
import re
import os
import sys

# 添加 hipporag 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'src'))
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 定义详细的情绪列表（基于Plutchik情绪轮和常见情绪）
EMOTIONS = [
    # 基本情绪（Plutchik的8种基本情绪）
    "joy",           # 快乐
    "sadness",       # 悲伤
    "anger",         # 愤怒
    "fear",          # 恐惧
    "surprise",      # 惊讶
    "disgust",       # 厌恶
    "trust",         # 信任
    "anticipation",  # 期待
    
    # 扩展情绪
    "love",          # 爱
    "hate",          # 恨
    "anxiety",       # 焦虑
    "calm",          # 平静
    "excitement",    # 兴奋
    "disappointment", # 失望
    "pride",         # 骄傲
    "shame",         # 羞耻
    "guilt",         # 愧疚
    "relief",        # 解脱
    "hope",          # 希望
    "despair",       # 绝望
    "contentment",   # 满足
    "frustration",   # 沮丧
    "gratitude",     # 感激
    "resentment",    # 怨恨
    "loneliness",    # 孤独
    "nostalgia",     # 怀旧
    "envy",          # 嫉妒
    "contempt",      # 轻蔑
]


def cosine_similarity(vec1, vec2):
    """
    计算两个情绪向量的余弦相似度（点积）
    
    注意：由于情绪向量所有分量都是非负值（0-1），归一化后仍在正象限，
    因此相似度范围是 [0, 1]，而不是 [-1, 1]
    - 1.0: 完全相同（所有情绪分布完全一致）
    - 0.0: 完全不同（没有共同的情绪）
    
    Args:
        vec1: 第一个情绪向量
        vec2: 第二个情绪向量
    
    Returns:
        float: 相似度值，范围 [0, 1]
    """
    return np.dot(vec1, vec2)


class EmotionExtractor:
    """情感向量提取器"""
    
    def __init__(self, api_key=None, api_base_url=None, model_name="GLM-4-Flash"):
        """
        初始化情感提取器
        
        Args:
            api_key: API 密钥，如果为 None 则从环境变量读取
            api_base_url: API 基础 URL，如果为 None 则从环境变量读取
            model_name: 使用的模型名称
        """
        # 从环境变量或参数获取配置
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            # 尝试从 Amygdala 配置读取
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
                from api_config import PARALLEL_API_KEY
                self.api_key = PARALLEL_API_KEY
            except ImportError:
                pass
        
        if not self.api_key:
            raise ValueError("API_KEY not found. Please set OPENAI_API_KEY or provide api_key parameter.")
        
        # API URL
        if api_base_url:
            self.api_url = f"{api_base_url.rstrip('/')}/chat/completions"
        else:
            api_base = os.getenv('OPENAI_API_BASE') or os.getenv('PARALLEL_BASE_URL')
            if api_base:
                self.api_url = f"{api_base.rstrip('/')}/chat/completions"
            else:
                try:
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
                    from api_config import PARALLEL_BASE_URL
                    self.api_url = f"{PARALLEL_BASE_URL.rstrip('/')}/chat/completions"
                except ImportError:
                    self.api_url = "https://llmapi.paratera.com/v1/chat/completions"
        
        self.model_name = model_name
        logger.info(f"EmotionExtractor initialized with model: {self.model_name}, API: {self.api_url}")
    
    def extract_emotion_vector(self, text):
        """
        从文本中提取情绪向量
        
        Args:
            text: 输入文本
        
        Returns:
            tuple: (numpy array, dict) - 归一化后的情绪向量和情绪字典
        """
        emotions_str = ", ".join(EMOTIONS)
        
        prompt = f"""You are an emotion analysis expert. Analyze the emotional content of the given text and assign intensity scores (0.0 to 1.0) for each emotion.

Emotion List:
{emotions_str}

Instructions:
1. Read the text carefully
2. For each emotion, assign a score from 0.0 (not present) to 1.0 (extremely strong)
3. Be precise - only assign high scores to emotions that are clearly present
4. Output ONLY a JSON object with emotion names as keys and scores as values
5. Do not include any explanation or additional text

Output Format (JSON only):
{{
  "joy": 0.8,
  "sadness": 0.1,
  "anger": 0.0,
  ...
}}

Text to analyze:
"{text}"

Output the JSON object only:"""

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,  # 低温度保证一致性
            "max_tokens": 500
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content'].strip()
            
            # 提取JSON（可能包含markdown代码块）
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
            
            try:
                emotion_dict = json.loads(json_str)
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试提取数字
                logger.warning(f"Failed to parse JSON, attempting to extract values from: {content[:200]}")
                emotion_dict = {}
                for emotion in EMOTIONS:
                    pattern = f'"{emotion}"\\s*:\\s*([0-9.]+)'
                    match = re.search(pattern, content)
                    if match:
                        emotion_dict[emotion] = float(match.group(1))
                    else:
                        emotion_dict[emotion] = 0.0
            
            # 构建向量（按照EMOTIONS的顺序）
            vector = np.array([emotion_dict.get(emotion, 0.0) for emotion in EMOTIONS])
            
            # L2归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            else:
                # 如果向量全为0，返回均匀分布
                vector = np.ones(len(EMOTIONS)) / len(EMOTIONS)
            
            return vector, emotion_dict
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise Exception(f"Chat API Error: {e}")
        except Exception as e:
            logger.error(f"Failed to extract emotion vector: {e}")
            raise
    
    def batch_extract_emotion_vectors(self, texts, batch_size=10):
        """
        批量提取情绪向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
        
        Returns:
            list: 情绪向量列表
        """
        vectors = []
        emotion_dicts = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing emotion batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch:
                try:
                    vector, emotion_dict = self.extract_emotion_vector(text)
                    vectors.append(vector)
                    emotion_dicts.append(emotion_dict)
                except Exception as e:
                    logger.error(f"Failed to extract emotion for text: {text[:50]}... Error: {e}")
                    # 返回零向量作为默认值
                    vectors.append(np.zeros(len(EMOTIONS)))
                    emotion_dicts.append({emotion: 0.0 for emotion in EMOTIONS})
        
        return vectors, emotion_dicts

