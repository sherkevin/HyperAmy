"""
Particle 模块 (H-Mem System V3)

基于庞加莱球的引力时间膨胀与热力学遗忘机制。
"""

# 核心模块 (V3)
from .properties import (
    ParticleProperties,
    PropertyCalculator,
    compute_properties,
    DEFAULT_ALPHA_MASS,
    DEFAULT_BETA_MASS,
    DEFAULT_T0,
    DEFAULT_RADIUS_SCALE,
)
from .memory_particle import (
    MemoryParticleV3,
    create_particle_from_emotion,
    create_particle_from_properties,
    ParticleUtils,
    DEFAULT_CURVATURE,
    DEFAULT_GAMMA,
)

__all__ = [
    # System V3 (新系统)
    'ParticleProperties',
    'PropertyCalculator',
    'compute_properties',
    'MemoryParticleV3',
    'create_particle_from_emotion',
    'create_particle_from_properties',
    'ParticleUtils',
    # V3 参数
    'DEFAULT_ALPHA_MASS',
    'DEFAULT_BETA_MASS',
    'DEFAULT_T0',
    'DEFAULT_RADIUS_SCALE',
    'DEFAULT_CURVATURE',
    'DEFAULT_GAMMA',
]

# 旧系统 (可选导入，计划废弃)
try:
    from .emotion_v2 import EmotionV2, EmotionNode
    from .speed import Speed
    from .temperature import Temperature
    from .particle import Particle, ParticleEntity

    __all__.extend([
        'EmotionV2',
        'EmotionNode',
        'Speed',
        'Temperature',
        'Particle',
        'ParticleEntity',
    ])
except Exception:
    # 旧系统可能缺少依赖（如 API_KEY），忽略错误
    pass
