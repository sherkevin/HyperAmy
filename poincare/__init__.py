"""
Poincare 模块

HyperAmy 的存储与混合检索模块，实现基于双曲空间的情绪记忆系统。

职责：
- 负责粒子在欧式空间和双曲空间的位置转换功能
- 提供检索功能和存储功能的调用层

注意：Point 类已移除，请使用 particle.ParticleEntity。
"""

from .types import SearchResult
from .physics import TimePhysics, ParticleProjector
from .storage import HyperAmyStorage
from .retrieval import HyperAmyRetrieval
from .linking import (
    build_hyperbolic_links,
    update_entities_with_links,
    auto_link_entities
)

# ParticleEntity 从 particle 模块导入，使用延迟导入避免循环依赖
# 注意：ParticleEntity 仅在运行时导入，避免在模块初始化时触发依赖链
# 用户应该直接从 particle 模块导入 ParticleEntity

__all__ = [
    'SearchResult',
    'TimePhysics',
    'ParticleProjector',
    'HyperAmyStorage',
    'HyperAmyRetrieval',
    'build_hyperbolic_links',
    'update_entities_with_links',
    'auto_link_entities',
]

__version__ = '1.2.0'

