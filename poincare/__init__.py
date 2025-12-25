"""
Poincare 模块

HyperAmy 的存储与混合检索模块，实现基于双曲空间的情绪记忆系统。
"""

from .types import Point, SearchResult
from .physics import TimePhysics, ParticleProjector
from .storage import HyperAmyStorage
from .retrieval import HyperAmyRetrieval
from .linking import (
    build_hyperbolic_links,
    update_points_with_links,
    auto_link_points
)

__all__ = [
    'Point',
    'SearchResult',
    'TimePhysics',
    'ParticleProjector',
    'HyperAmyStorage',
    'HyperAmyRetrieval',
    'build_hyperbolic_links',
    'update_points_with_links',
    'auto_link_points',
]

__version__ = '1.2.0'

