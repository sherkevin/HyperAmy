"""
Amygdala 情感分析模块

基于 HyperAmy 的情感分析功能，集成到 HippoRAG 框架中。
"""

from .sentiment_vector import sentimentExtractor, sentimentS, cosine_similarity
from .sentiment_store import sentimentStore

__all__ = [
    'sentimentExtractor',
    'sentimentStore',
    'sentimentS',
    'cosine_similarity',
]

