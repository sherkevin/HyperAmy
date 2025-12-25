"""
Particle 模块

简洁的情感向量提取模块：输入 chunk，输出 emotion vector
"""

from .emotion import Emotion
from .emotion_v2 import EmotionV2, EmotionNode
from .particle import Particle, ParticleProperty

__all__ = ['Emotion', 'EmotionV2', 'EmotionNode', 'Particle', 'ParticleProperty']

