"""
ParticleProjector - 向后兼容适配器

提供与旧系统兼容的 ParticleProjector 接口，
内部使用 V3 的 PhysicsEngine 实现。
"""
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from poincare.physics import PhysicsEngine
from poincare.math import project_to_poincare, extract_radius
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectedState:
    """投影后的粒子状态（兼容旧接口）"""
    entity_id: str
    poincare_coord: np.ndarray
    hyperbolic_radius: float
    conversation_id: str
    metadata: Dict[str, Any]


class ParticleProjector:
    """
    粒子投影器（向后兼容适配器）

    适配旧系统的 ParticleProjector 接口，内部使用 V3 PhysicsEngine。

    旧接口参数：
    - curvature: 空间曲率 c
    - scaling_factor: 缩放因子
    - max_radius: 最大半径

    V3 实现：
    - 使用 PhysicsEngine 进行状态计算
    - 使用 project_to_poincare 进行坐标投影
    """

    def __init__(
        self,
        curvature: float = 1.0,
        scaling_factor: float = 2.0,
        max_radius: float = 10000.0
    ):
        """
        初始化粒子投影器

        Args:
            curvature: 空间曲率 c
            scaling_factor: 缩放因子（用于计算初始半径）
            max_radius: 最大半径（用于限制投影范围）
        """
        self.curvature = curvature
        self.scaling_factor = scaling_factor
        self.max_radius = max_radius

        # 使用 V3 PhysicsEngine
        self.physics = PhysicsEngine(curvature=curvature, gamma=1.0)

        logger.info(
            f"ParticleProjector initialized (V3 adapter): c={curvature}, "
            f"scaling={scaling_factor}, max_radius={max_radius}"
        )

    def project(
        self,
        emotion_vector: np.ndarray,
        speed: float,
        weight: float,
        temperature: float,
        born: float,
        entity_id: str,
        conversation_id: str,
        t_now: Optional[float] = None,
        **metadata
    ) -> ProjectedState:
        """
        将粒子投影到庞加莱球（兼容旧接口）

        Args:
            emotion_vector: 情感向量（语义方向）
            speed: 速度（影响初始半径）
            weight: 质量
            temperature: 温度
            born: 创建时间
            entity_id: 实体 ID
            conversation_id: 对话 ID
            t_now: 当前时间
            **metadata: 额外元数据

        Returns:
            ProjectedState 对象
        """
        import time

        if t_now is None:
            t_now = time.time()

        # 计算初始半径（使用 scaling_factor）
        initial_radius = self.scaling_factor * weight

        # 归一化方向向量
        direction = emotion_vector / np.linalg.norm(emotion_vector)

        # 使用 V3 PhysicsEngine 计算状态
        state = self.physics.compute_state(
            direction=direction,
            mass=weight,
            temperature=temperature,
            initial_radius=initial_radius,
            created_at=born,
            t_now=t_now
        )

        # 限制最大半径
        if state.hyperbolic_radius > self.max_radius:
            state.hyperbolic_radius = self.max_radius
            state.poincare_coord = project_to_poincare(
                direction, state.hyperbolic_radius, self.curvature
            )

        return ProjectedState(
            entity_id=entity_id,
            poincare_coord=state.poincare_coord,
            hyperbolic_radius=state.hyperbolic_radius,
            conversation_id=conversation_id,
            metadata={
                'direction': direction,
                'mass': weight,
                'temperature': temperature,
                'speed': speed,
                'born': born,
                'initial_radius': initial_radius,
                'memory_strength': state.memory_strength,
                'is_forgotten': state.is_forgotten,
                **metadata
            }
        )

    def compute_batch(
        self,
        particles: List[Dict[str, Any]],
        t_now: Optional[float] = None
    ) -> List[ProjectedState]:
        """
        批量投影粒子

        Args:
            particles: 粒子字典列表
            t_now: 当前时间

        Returns:
            ProjectedState 列表
        """
        results = []
        for p in particles:
            state = self.project(
                emotion_vector=p.get('emotion_vector'),
                speed=p.get('speed', 0.5),
                weight=p.get('weight', 1.0),
                temperature=p.get('temperature', 1.0),
                born=p.get('born', 0),
                entity_id=p.get('entity_id', ''),
                conversation_id=p.get('conversation_id', ''),
                t_now=t_now,
                **{k: v for k, v in p.items() if k not in
                   ['emotion_vector', 'speed', 'weight', 'temperature', 'born', 'entity_id', 'conversation_id']}
            )
            results.append(state)
        return results


__all__ = ['ParticleProjector', 'ProjectedState']
