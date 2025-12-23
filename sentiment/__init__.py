"""
Amygdala 情感分析模块

基于 HyperAmy 的情感分析功能，集成到 HippoRAG 框架中。
"""

from .emotion_vector import EmotionExtractor, EMOTIONS, cosine_similarity
from .emotion_store import EmotionStore

__all__ = [
    'EmotionExtractor',
    'EmotionStore',
    'EMOTIONS',
    'cosine_similarity',
]

