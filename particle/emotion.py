"""
Emotion 类

简洁的情感向量提取：输入 chunk，输出 emotion vector
"""
import numpy as np
import json
import re

from llm import create_client
from llm.config import API_URL_CHAT, DEFAULT_MODEL
from hipporag.utils.logging_utils import get_logger
from prompts.prompt_template_manager import PromptTemplateManager

logger = get_logger(__name__)

# 定义情绪列表（30种情绪）
# EMOTIONS = [
#     # 基本情绪（Plutchik的8种基本情绪）
#     "joy",           # 快乐
#     "sadness",       # 悲伤
#     "anger",         # 愤怒
#     "fear",          # 恐惧
#     "surprise",      # 惊讶
#     "disgust",       # 厌恶
#     "trust",         # 信任
#     "anticipation",  # 期待
    
#     # 扩展情绪
#     "love",          # 爱
#     "hate",          # 恨
#     "anxiety",       # 焦虑
#     "calm",          # 平静
#     "excitement",    # 兴奋
#     "disappointment", # 失望
#     "pride",         # 骄傲
#     "shame",         # 羞耻
#     "guilt",         # 愧疚
#     "relief",         # 解脱
#     "hope",          # 希望
#     "despair",       # 绝望
#     "contentment",   # 满足
#     "frustration",   # 沮丧
#     "gratitude",     # 感激
#     "resentment",    # 怨恨
#     "loneliness",    # 孤独
#     "nostalgia",     # 怀旧
#     "envy",          # 嫉妒
#     "contempt",      # 轻蔑
# ]

EMOTIONS = [
    # Positive
    "admiration", "amusement", "approval", "caring", "desire", 
    "excitement", "gratitude", "joy", "love", "optimism", 
    "pride", "relief",
    
    # Negative
    "anger", "annoyance", "disappointment", "disapproval", "disgust", 
    "embarrassment", "fear", "grief", "nervousness", "remorse", 
    "sadness", 
    
    # Ambiguous / Cognitive
    "confusion", "curiosity", "realization", "surprise",
    
    # Neutral
    "neutral"
]


class Emotion:
    """
    情感向量提取类
    
    功能：输入 chunk，输出 emotion vector
    """
    
    def __init__(self, model_name=None):
        """
        初始化 Emotion 类
        
        Args:
            model_name: 使用的模型名称，如果为 None 则使用默认模型
        """
        self.model_name = model_name or DEFAULT_MODEL
        
        # 创建 LLM 客户端（使用 normal 模式，Chat API）
        self.client = create_client(
            model_name=self.model_name,
            chat_api_url=API_URL_CHAT,
            mode="normal"
        )
        
        # 初始化 prompt 模板管理器
        self.prompt_template_manager = PromptTemplateManager()
        
        logger.info(f"Emotion initialized with model: {self.model_name}")
    
    def extract(self, chunk: str) -> np.ndarray:
        """
        提取 chunk 的情感向量
        
        Args:
            chunk: 输入文本片段
        
        Returns:
            numpy.ndarray: 归一化后的情感向量 (30维)
        """
        emotions_str = ", ".join(EMOTIONS)
        
        # 使用模板管理器渲染 prompt
        prompt = self.prompt_template_manager.render(
            name='emotion_extraction',
            emotions_list=emotions_str,
            chunk=chunk
        )

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
            
            return vector
            
        except Exception as e:
            logger.error(f"Failed to extract emotion vector: {e}")
            raise

