"""
Particle 模块

粒子处理模块：整合 emotion、speed、temperature 的逻辑。
输入文本和id，输出包含实体属性列表。
"""

from .emotion_v2 import EmotionV2, EmotionNode
from .speed import Speed
from .temperature import Temperature
from .particle import Particle, ParticleEntity

__all__ = [
    'EmotionV2',
    'EmotionNode',
    'Speed',
    'Temperature',
    'Particle',
    'ParticleEntity',
]

