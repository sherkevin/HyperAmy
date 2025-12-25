"""
情感向量提取模块

从 HyperAmy 提取的核心功能，用于提取文本的情感向量。
"""
import numpy as np
import json
import re
import os

from llm import create_client
from llm.config import API_URL_CHAT, DEFAULT_MODEL
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 定义详细的情绪列表（基于Plutchik情绪轮和常见情绪）
sentimentS = [
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


class sentimentExtractor:
    """情感向量提取器"""
    
    def __init__(self, api_key=None, api_base_url=None, model_name=None):
        """
        初始化情感提取器
        
        Args:
            api_key: API 密钥，如果为 None 则从 llm.config 读取
            api_base_url: API 基础 URL，如果为 None 则从 llm.config 读取
            model_name: 使用的模型名称，如果为 None 则使用默认模型
        """
        # 使用 llm.config 和 create_client 创建客户端
        self.model_name = model_name or DEFAULT_MODEL
        
        # 创建 LLM 客户端（使用 normal 模式，Chat API）
        self.client = create_client(
            api_key=api_key,
            model_name=self.model_name,
            chat_api_url=api_base_url or API_URL_CHAT,
            mode="normal"
        )
        
        logger.info(f"sentimentExtractor initialized with model: {self.model_name}")
    
    def extract_sentiment_vector(self, text):
        """
        从文本中提取情绪向量
        
        Args:
            text: 输入文本
        
        Returns:
            tuple: (numpy array, dict) - 归一化后的情绪向量和情绪字典
        """
        sentiments_str = ", ".join(sentimentS)
        
        prompt = f"""You are an sentiment analysis expert. Analyze the sentimental content of the given text and assign intensity scores (0.0 to 1.0) for each sentiment.

sentiment List:
{sentiments_str}

Instructions:
1. Read the text carefully
2. For each sentiment, assign a score from 0.0 (not present) to 1.0 (extremely strong)
3. Be precise - only assign high scores to sentiments that are clearly present
4. Output ONLY a JSON object with sentiment names as keys and scores as values
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

        try:
            # 使用 CompletionClient 调用 API
            result = self.client.complete(
                query=prompt,
                max_tokens=500,
                temperature=0.2  # 低温度保证一致性
            )
            
            content = result.get_answer_text().strip()
            
            # 提取JSON（可能包含markdown代码块）
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
            
            try:
                sentiment_dict = json.loads(json_str)
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试提取数字
                logger.warning(f"Failed to parse JSON, attempting to extract values from: {content[:200]}")
                sentiment_dict = {}
                for sentiment in sentimentS:
                    pattern = f'"{sentiment}"\\s*:\\s*([0-9.]+)'
                    match = re.search(pattern, content)
                    if match:
                        sentiment_dict[sentiment] = float(match.group(1))
                    else:
                        sentiment_dict[sentiment] = 0.0
            
            # 构建向量（按照sentimentS的顺序）
            vector = np.array([sentiment_dict.get(sentiment, 0.0) for sentiment in sentimentS])
            
            # L2归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            else:
                # 如果向量全为0，返回均匀分布
                vector = np.ones(len(sentimentS)) / len(sentimentS)
            
            return vector, sentiment_dict
            
        except Exception as e:
            logger.error(f"Failed to extract sentiment vector: {e}")
            raise
    
    def batch_extract_sentiment_vectors(self, texts, batch_size=10):
        """
        批量提取情绪向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
        
        Returns:
            list: 情绪向量列表
        """
        vectors = []
        sentiment_dicts = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing sentiment batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch:
                try:
                    vector, sentiment_dict = self.extract_sentiment_vector(text)
                    vectors.append(vector)
                    sentiment_dicts.append(sentiment_dict)
                except Exception as e:
                    logger.error(f"Failed to extract sentiment for text: {text[:50]}... Error: {e}")
                    # 返回零向量作为默认值
                    vectors.append(np.zeros(len(sentimentS)))
                    sentiment_dicts.append({sentiment: 0.0 for sentiment in sentimentS})
        
        return vectors, sentiment_dicts

