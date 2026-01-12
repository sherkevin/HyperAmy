"""
Poincare 模块 (H-Mem System V3)

基于庞加莱球的引力时间膨胀与热力学遗忘机制。

核心模块：
- math: 庞加莱球基础几何运算
- physics: 物理引擎（引力时间膨胀）
- storage: 存储层
- retrieval: 检索层（三步检索流程）
"""

# V3 核心模块
from .math import (
    PoincareBall,
    conformal_factor,
    mobius_add,
    poincare_dist,
    project_to_poincare,
    extract_radius,
)
from .physics import (
    ParticleState,
    TimeDynamics,
    FastPositionUpdate,
    PhysicsEngine,
    compute_particle_state,
)
from .types import SearchResult
from .storage import HyperAmyStorage

# 向后兼容：尝试导入旧系统组件
try:
    from .linking import (
        build_hyperbolic_links,
        update_entities_with_links,
        auto_link_entities
    )
    _has_linking = True
except Exception:
    _has_linking = False

    # 创建占位符函数以避免导入错误
    def auto_link_entities(*args, **kwargs):
        """占位符函数：链接功能正在重构中"""
        return []

    def build_hyperbolic_links(*args, **kwargs):
        """占位符函数：链接功能正在重构中"""
        return {}

    def update_entities_with_links(*args, **kwargs):
        """占位符函数：链接功能正在重构中"""
        pass

# 向后兼容：ParticleProjector
try:
    from .projector import ParticleProjector
except Exception:
    # 使用 physics 模块作为替代
    ParticleProjector = PhysicsEngine

__all__ = [
    # V3 核心类
    'PoincareBall',
    'ParticleState',
    'TimeDynamics',
    'FastPositionUpdate',
    'PhysicsEngine',
    # V3 函数
    'conformal_factor',
    'mobius_add',
    'poincare_dist',
    'project_to_poincare',
    'extract_radius',
    'compute_particle_state',
    # 其他
    'SearchResult',
    'HyperAmyStorage',
    # 向后兼容
    'ParticleProjector',
    'auto_link_entities',
    'build_hyperbolic_links',
    'update_entities_with_links',
]

__version__ = '3.0.0'
