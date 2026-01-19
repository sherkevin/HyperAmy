"""
Memory Particle Module (H-Mem System V3)

记忆粒子类，实现基于引力时间膨胀的遗忘机制。

根据 system_v3.md 设计文档：
- 粒子沿测地线向球心坠落（半径减小）
- R(t) = R₀ · exp(-γ/m · Δt)
- 质量在分母：质量越大，衰减越慢

物理隐喻：
- 边界 (||z|| → 1/√c): Event Horizon of Clarity (极致清晰)
- 球心 (z = 0): Singularity of Oblivion (遗忘奇点/黑洞)

Author: HyperAmy Team
Version: 3.0
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from poincare.math import PoincareBall, project_to_poincare, extract_radius
from particle.properties import ParticleProperties, PropertyCalculator
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 默认参数
DEFAULT_CURVATURE = 1.0       # 庞加莱球曲率 c
DEFAULT_GAMMA = 1.0           # 宇宙衰变常数
DEFAULT_RADIUS_SCALE = 1.0    # 初始半径缩放系数


@dataclass
class MemoryParticleV3:
    """
    记忆粒子 (System V3)

    粒子在庞加莱球中沿测地线向球心坠落，模拟遗忘过程。

    Attributes:
        direction: 语义方向 μ (归一化向量)
        mass: 引力质量 m
        temperature: 热力学温度 T
        initial_radius: 初始双曲半径 R₀
        created_at: 创建时间戳 t₀
        curvature: 空间曲率 c
        gamma: 衰变常数 γ
    """
    direction: np.ndarray     # 语义方向 μ
    mass: float               # 引力质量 m
    temperature: float        # 热力学温度 T
    initial_radius: float     # 初始双曲半径 R₀
    created_at: float         # 创建时间戳 t₀
    curvature: float = DEFAULT_CURVATURE  # 曲率 c
    gamma: float = DEFAULT_GAMMA          # 衰变常数 γ

    def __post_init__(self):
        """初始化后处理"""
        # 确保方向是归一化的
        norm = np.linalg.norm(self.direction)
        if norm > 1e-9:
            self.direction = self.direction / norm

        # 创建庞加莱球空间
        self._space = PoincareBall(curvature=self.curvature, dimension=len(self.direction))

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return len(self.direction)

    def age(self, t_now: Optional[float] = None) -> float:
        """
        获取粒子年龄

        Args:
            t_now: 当前时间戳（默认为系统时间）

        Returns:
            年龄 Δt = t_now - t₀
        """
        if t_now is None:
            t_now = time.time()
        return max(0.0, t_now - self.created_at)

    def hyperbolic_radius(self, t_now: Optional[float] = None) -> float:
        """
        计算当前双曲半径 R(t)

        公式: R(t) = R₀ · exp(-γ/m · Δt)

        物理意义：
        - 质量在分母：质量越大，衰减越慢
        - γ/m 是时间膨胀因子

        Args:
            t_now: 当前时间戳（默认为系统时间）

        Returns:
            当前双曲半径 R(t)
        """
        dt = self.age(t_now)

        # 引力时间膨胀衰变公式
        decay_factor = math.exp(-self.gamma / self.mass * dt)
        R_t = self.initial_radius * decay_factor

        return float(R_t)

    def poincare_coord(self, t_now: Optional[float] = None) -> np.ndarray:
        """
        计算当前庞加莱坐标 z(t)

        公式: z(t) = tanh(√c/2 · R(t)) · μ

        在庞加莱球模型中，从原点出发的测地线在欧氏空间中是直线，
        所以粒子沿径向运动，方向 μ 不变。

        Args:
            t_now: 当前时间戳（默认为系统时间）

        Returns:
            庞加莱坐标 z(t)
        """
        R_t = self.hyperbolic_radius(t_now)

        # 投影到庞加莱球
        z_t = project_to_poincare(self.direction, R_t, self.curvature)

        # 转换为 numpy
        if isinstance(z_t, np.ndarray):
            return z_t
        else:
            return z_t.detach().cpu().numpy()

    def is_forgotten(self, t_now: Optional[float] = None, threshold: float = 1e-3) -> bool:
        """
        判断粒子是否已被遗忘（落入黑洞）

        Args:
            t_now: 当前时间戳
            threshold: 双曲半径阈值

        Returns:
            True 如果粒子已被遗忘
        """
        R_t = self.hyperbolic_radius(t_now)
        return R_t < threshold

    def euclidean_norm(self, t_now: Optional[float] = None) -> float:
        """
        计算当前欧氏坐标的模长 ||z(t)||

        Args:
            t_now: 当前时间戳

        Returns:
            欧氏模长
        """
        z_t = self.poincare_coord(t_now)
        return float(np.linalg.norm(z_t))

    def time_to_forget(self, threshold: float = 1e-3, t_now: Optional[float] = None) -> float:
        """
        计算粒子被遗忘所需的时间

        由 R(t) = R₀ · exp(-γ/m · t) = threshold
        解得: t = (m/γ) · ln(R₀/threshold)

        Args:
            threshold: 遗忘阈值
            t_now: 当前时间戳

        Returns:
            距离遗忘的剩余时间（秒）
        """
        R_t = self.hyperbolic_radius(t_now)
        if R_t <= threshold:
            return 0.0

        remaining_time = (self.mass / self.gamma) * math.log(R_t / threshold)
        return float(remaining_time)

    def memory_strength(self, t_now: Optional[float] = None) -> float:
        """
        计算记忆强度（归一化的双曲半径）

        Returns:
            归一化强度 [0, 1]，1 为初始强度，0 为完全遗忘
        """
        R_t = self.hyperbolic_radius(t_now)
        return float(R_t / self.initial_radius)

    def __repr__(self) -> str:
        return (
            f"MemoryParticleV3("
            f"m={self.mass:.4f}, "
            f"T={self.temperature:.4f}, "
            f"R₀={self.initial_radius:.4f}, "
            f"R(t)={self.hyperbolic_radius():.4f}, "
            f"age={self.age():.2f}s)"
        )


def create_particle_from_emotion(
    emotion_vector: np.ndarray,
    curvature: float = DEFAULT_CURVATURE,
    gamma: float = DEFAULT_GAMMA,
    alpha_mass: float = 1.0,
    beta_mass: float = 1.0,
    T0: float = 1.0,
    radius_scale: float = DEFAULT_RADIUS_SCALE,
    t_created: Optional[float] = None
) -> MemoryParticleV3:
    """
    从情绪向量创建记忆粒子

    Args:
        emotion_vector: 情绪向量（未归一化）
        curvature: 空间曲率 c
        gamma: 衰变常数 γ
        alpha_mass: 质量公式系数 α
        beta_mass: 质量公式系数 β
        T0: 基准温度
        radius_scale: 半径缩放系数
        t_created: 创建时间戳（默认为当前时间）

    Returns:
        MemoryParticleV3 实例
    """
    # 计算粒子属性
    calc = PropertyCalculator(
        alpha_mass=alpha_mass,
        beta_mass=beta_mass,
        T0=T0,
        radius_scale=radius_scale
    )
    props = calc.compute_properties(emotion_vector)

    # 创建粒子
    if t_created is None:
        t_created = time.time()

    particle = MemoryParticleV3(
        direction=props.direction,
        mass=props.mass,
        temperature=props.temperature,
        initial_radius=props.initial_radius,
        created_at=t_created,
        curvature=curvature,
        gamma=gamma
    )

    logger.debug(
        f"Created particle: m={props.mass:.4f}, T={props.temperature:.4f}, "
        f"R₀={props.initial_radius:.4f}"
    )

    return particle


def create_particle_from_properties(
    direction: np.ndarray,
    mass: float,
    temperature: float,
    initial_radius: float,
    created_at: Optional[float] = None,
    curvature: float = DEFAULT_CURVATURE,
    gamma: float = DEFAULT_GAMMA
) -> MemoryParticleV3:
    """
    从已有属性创建记忆粒子

    Args:
        direction: 语义方向 μ
        mass: 引力质量 m
        temperature: 热力学温度 T
        initial_radius: 初始双曲半径 R₀
        created_at: 创建时间戳（默认为当前时间）
        curvature: 空间曲率 c
        gamma: 衰变常数 γ

    Returns:
        MemoryParticleV3 实例
    """
    if created_at is None:
        created_at = time.time()

    return MemoryParticleV3(
        direction=direction,
        mass=mass,
        temperature=temperature,
        initial_radius=initial_radius,
        created_at=created_at,
        curvature=curvature,
        gamma=gamma
    )


class ParticleUtils:
    """粒子工具类"""

    @staticmethod
    def decay_curve(
        particle: MemoryParticleV3,
        time_points: list,
        t_start: float = 0.0
    ) -> dict:
        """
        计算粒子的衰减曲线

        Args:
            particle: 记忆粒子
            time_points: 时间点列表
            t_start: 起始时间偏移

        Returns:
            包含 time, radius, strength 的字典
        """
        # 创建临时粒子，用于模拟
        sim_particle = MemoryParticleV3(
            direction=particle.direction.copy(),
            mass=particle.mass,
            temperature=particle.temperature,
            initial_radius=particle.initial_radius,
            created_at=t_start,
            curvature=particle.curvature,
            gamma=particle.gamma
        )

        times = []
        radii = []
        strengths = []

        for dt in time_points:
            t_now = t_start + dt
            times.append(dt)
            radii.append(sim_particle.hyperbolic_radius(t_now))
            strengths.append(sim_particle.memory_strength(t_now))

        return {
            'time': times,
            'radius': radii,
            'strength': strengths
        }

    @staticmethod
    def compare_decay(particles: list, time_points: list) -> dict:
        """
        比较多个粒子的衰减曲线

        Args:
            particles: 粒子列表
            time_points: 时间点列表

        Returns:
            每个粒子的衰减曲线字典
        """
        results = {}
        for i, p in enumerate(particles):
            results[f'particle_{i}'] = ParticleUtils.decay_curve(p, time_points)
        return results


# 导出
__all__ = [
    'MemoryParticleV3',
    'create_particle_from_emotion',
    'create_particle_from_properties',
    'ParticleUtils',
    'DEFAULT_CURVATURE',
    'DEFAULT_GAMMA',
    'DEFAULT_RADIUS_SCALE',
]
