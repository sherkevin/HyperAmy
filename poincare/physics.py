"""
Physics Engine Module (H-Mem System V3)

物理引擎：实现引力时间膨胀与遗忘机制

根据 system_v3.md 设计文档：
- 粒子沿测地线向球心坠落（半径减小）
- R(t) = R₀ · exp(-γ/m · Δt)
- 质量在分母：质量越大，衰减越慢（引力时间膨胀）

核心物理定律：
1. TimeDynamics: 引力时间膨胀衰变函数
2. FastPositionUpdate: O(1) 复杂度的位置更新

Author: HyperAmy Team
Version: 3.0
"""

import math
import time
import numpy as np
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass

from poincare.math import project_to_poincare, extract_radius
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 默认参数
DEFAULT_CURVATURE = 1.0       # 庞加莱球曲率 c
DEFAULT_GAMMA = 1.0           # 宇宙衰变常数
FORGETTING_THRESHOLD = 1e-3  # 遗忘阈值


@dataclass
class ParticleState:
    """
    粒子在某一时刻的完整状态

    Attributes:
        direction: 语义方向 μ
        mass: 引力质量 m
        temperature: 热力学温度 T
        hyperbolic_radius: 当前双曲半径 R(t)
        poincare_coord: 庞加莱坐标 z(t)
        memory_strength: 记忆强度 (R(t) / R₀)
        is_forgotten: 是否被遗忘
    """
    direction: np.ndarray
    mass: float
    temperature: float
    hyperbolic_radius: float
    poincare_coord: np.ndarray
    memory_strength: float
    is_forgotten: bool

    def __repr__(self) -> str:
        return (
            f"ParticleState(R={self.hyperbolic_radius:.4f}, "
            f"strength={self.memory_strength:.4f}, "
            f"forgotten={self.is_forgotten})"
        )


class TimeDynamics:
    """
    时间动力学：引力时间膨胀衰变函数

    核心公式：R(t) = R₀ · exp(-γ/m · Δt)

    物理意义：
    - γ/m 是时间膨胀因子
    - 质量在分母：质量越大，时间流逝越慢，衰减越慢
    - Trauma (m → ∞): 衰减因子 → 1，记忆永不磨灭
    - Trivia (m → 0): 衰减因子 → 0，记忆瞬间消失
    """

    @staticmethod
    def hyperbolic_radius(
        initial_radius: float,
        mass: float,
        delta_t: float,
        gamma: float = DEFAULT_GAMMA
    ) -> float:
        """
        计算时刻 t 的双曲半径 R(t)

        公式: R(t) = R₀ · exp(-γ/m · Δt)

        Args:
            initial_radius: 初始双曲半径 R₀
            mass: 引力质量 m
            delta_t: 时间差 Δt = t - t₀
            gamma: 衰变常数 γ

        Returns:
            当前双曲半径 R(t)
        """
        if mass <= 0:
            return 0.0

        # 引力时间膨胀衰变
        decay_factor = math.exp(-gamma / mass * delta_t)
        return initial_radius * decay_factor

    @staticmethod
    def memory_strength(
        initial_radius: float,
        mass: float,
        delta_t: float,
        gamma: float = DEFAULT_GAMMA
    ) -> float:
        """
        计算归一化记忆强度

        公式: strength = R(t) / R₀ = exp(-γ/m · Δt)

        Args:
            initial_radius: 初始双曲半径 R₀
            mass: 引力质量 m
            delta_t: 时间差 Δt
            gamma: 衰变常数 γ

        Returns:
            归一化强度 [0, 1]
        """
        if initial_radius <= 0:
            return 0.0

        R_t = TimeDynamics.hyperbolic_radius(initial_radius, mass, delta_t, gamma)
        return R_t / initial_radius

    @staticmethod
    def time_to_forget(
        initial_radius: float,
        mass: float,
        threshold: float = FORGETTING_THRESHOLD,
        gamma: float = DEFAULT_GAMMA
    ) -> float:
        """
        计算遗忘所需的时间

        由 R(t) = R₀ · exp(-γ/m · t) = threshold
        解得: t = (m/γ) · ln(R₀/threshold)

        Args:
            initial_radius: 初始双曲半径 R₀
            mass: 引力质量 m
            threshold: 遗忘阈值
            gamma: 衰变常数 γ

        Returns:
            遗忘所需时间（秒）
        """
        if initial_radius <= threshold:
            return 0.0
        if mass <= 0:
            return 0.0

        return (mass / gamma) * math.log(initial_radius / threshold)

    @staticmethod
    def is_forgotten(
        initial_radius: float,
        mass: float,
        delta_t: float,
        threshold: float = FORGETTING_THRESHOLD,
        gamma: float = DEFAULT_GAMMA
    ) -> bool:
        """
        判断粒子是否已被遗忘

        Args:
            initial_radius: 初始双曲半径 R₀
            mass: 引力质量 m
            delta_t: 时间差 Δt
            threshold: 遗忘阈值
            gamma: 衰变常数 γ

        Returns:
            True 如果已被遗忘
        """
        R_t = TimeDynamics.hyperbolic_radius(initial_radius, mass, delta_t, gamma)
        return R_t < threshold


class FastPositionUpdate:
    """
    快速位置更新：O(1) 复杂度的庞加莱坐标计算

    在庞加莱球模型中，从原点出发的测地线在欧氏空间中是直线，
    所以粒子沿径向运动，方向 μ 不变。

    核心公式：z(t) = tanh(√c/2 · R(t)) · μ
    """

    @staticmethod
    def poincare_coord(
        direction: np.ndarray,
        hyperbolic_radius: float,
        curvature: float = DEFAULT_CURVATURE
    ) -> np.ndarray:
        """
        计算庞加莱坐标

        公式: z = tanh(√c/2 · R) · μ

        Args:
            direction: 语义方向 μ（归一化）
            hyperbolic_radius: 双曲半径 R
            curvature: 空间曲率 c

        Returns:
            庞加莱坐标 z
        """
        result = project_to_poincare(direction, hyperbolic_radius, curvature)
        if isinstance(result, np.ndarray):
            return result
        else:
            return result.detach().cpu().numpy()

    @staticmethod
    def compute_state(
        direction: np.ndarray,
        mass: float,
        temperature: float,
        initial_radius: float,
        created_at: float,
        t_now: float,
        curvature: float = DEFAULT_CURVATURE,
        gamma: float = DEFAULT_GAMMA,
        forgetting_threshold: float = FORGETTING_THRESHOLD
    ) -> ParticleState:
        """
        计算粒子在当前时刻的完整状态

        这是 O(1) 复杂度的快速位置解算，无需微分方程积分。

        Args:
            direction: 语义方向 μ
            mass: 引力质量 m
            temperature: 热力学温度 T
            initial_radius: 初始双曲半径 R₀
            created_at: 创建时间 t₀
            t_now: 当前时间 t
            curvature: 空间曲率 c
            gamma: 衰变常数 γ
            forgetting_threshold: 遗忘阈值

        Returns:
            ParticleState 对象
        """
        # 时间差
        delta_t = max(0.0, t_now - created_at)

        # 计算双曲半径
        R_t = TimeDynamics.hyperbolic_radius(initial_radius, mass, delta_t, gamma)

        # 计算庞加莱坐标
        z_t = FastPositionUpdate.poincare_coord(direction, R_t, curvature)

        # 记忆强度
        strength = TimeDynamics.memory_strength(initial_radius, mass, delta_t, gamma)

        # 是否遗忘
        forgotten = R_t < forgetting_threshold

        return ParticleState(
            direction=direction,
            mass=mass,
            temperature=temperature,
            hyperbolic_radius=R_t,
            poincare_coord=z_t,
            memory_strength=strength,
            is_forgotten=forgotten
        )


class PhysicsEngine:
    """
    物理引擎：统一的时间演化和位置更新接口

    整合 TimeDynamics 和 FastPositionUpdate，
    提供简洁的 API 用于计算粒子状态。
    """

    def __init__(
        self,
        curvature: float = DEFAULT_CURVATURE,
        gamma: float = DEFAULT_GAMMA,
        forgetting_threshold: float = FORGETTING_THRESHOLD
    ):
        """
        初始化物理引擎

        Args:
            curvature: 空间曲率 c
            gamma: 衰变常数 γ
            forgetting_threshold: 遗忘阈值
        """
        self.curvature = curvature
        self.gamma = gamma
        self.forgetting_threshold = forgetting_threshold

        logger.info(
            f"PhysicsEngine initialized: c={curvature}, γ={gamma}, "
            f"threshold={forgetting_threshold}"
        )

    def compute_state(
        self,
        direction: np.ndarray,
        mass: float,
        temperature: float,
        initial_radius: float,
        created_at: float,
        t_now: Optional[float] = None
    ) -> ParticleState:
        """
        计算粒子在当前时刻的完整状态

        Args:
            direction: 语义方向 μ
            mass: 引力质量 m
            temperature: 热力学温度 T
            initial_radius: 初始双曲半径 R₀
            created_at: 创建时间 t₀
            t_now: 当前时间（默认为系统时间）

        Returns:
            ParticleState 对象
        """
        if t_now is None:
            t_now = time.time()

        return FastPositionUpdate.compute_state(
            direction=direction,
            mass=mass,
            temperature=temperature,
            initial_radius=initial_radius,
            created_at=created_at,
            t_now=t_now,
            curvature=self.curvature,
            gamma=self.gamma,
            forgetting_threshold=self.forgetting_threshold
        )

    def compute_batch_states(
        self,
        particles: list,
        t_now: Optional[float] = None
    ) -> list:
        """
        批量计算粒子状态

        Args:
            particles: 粒子列表，每个粒子是 (direction, mass, temp, R0, created_at) 元组
            t_now: 当前时间（默认为系统时间）

        Returns:
            ParticleState 列表
        """
        if t_now is None:
            t_now = time.time()

        states = []
        for direction, mass, temp, R0, created_at in particles:
            state = self.compute_state(direction, mass, temp, R0, created_at, t_now)
            states.append(state)

        return states

    def hyperbolic_radius(
        self,
        initial_radius: float,
        mass: float,
        delta_t: float
    ) -> float:
        """计算双曲半径 R(t)"""
        return TimeDynamics.hyperbolic_radius(
            initial_radius, mass, delta_t, self.gamma
        )

    def is_forgotten(
        self,
        initial_radius: float,
        mass: float,
        delta_t: float
    ) -> bool:
        """判断是否遗忘"""
        return TimeDynamics.is_forgotten(
            initial_radius, mass, delta_t, self.forgetting_threshold, self.gamma
        )

    def time_to_forget(
        self,
        initial_radius: float,
        mass: float
    ) -> float:
        """计算遗忘时间"""
        return TimeDynamics.time_to_forget(
            initial_radius, mass, self.forgetting_threshold, self.gamma
        )


# 便捷函数
def compute_particle_state(
    direction: np.ndarray,
    mass: float,
    temperature: float,
    initial_radius: float,
    created_at: float,
    t_now: Optional[float] = None,
    curvature: float = DEFAULT_CURVATURE,
    gamma: float = DEFAULT_GAMMA
) -> ParticleState:
    """
    便捷函数：计算粒子状态（使用默认物理引擎）

    Args:
        direction: 语义方向 μ
        mass: 引力质量 m
        temperature: 热力学温度 T
        initial_radius: 初始双曲半径 R₀
        created_at: 创建时间 t₀
        t_now: 当前时间
        curvature: 空间曲率 c
        gamma: 衰变常数 γ

    Returns:
        ParticleState 对象
    """
    engine = PhysicsEngine(curvature=curvature, gamma=gamma)
    return engine.compute_state(direction, mass, temperature, initial_radius, created_at, t_now)


# 导出
__all__ = [
    'ParticleState',
    'TimeDynamics',
    'FastPositionUpdate',
    'PhysicsEngine',
    'compute_particle_state',
    'DEFAULT_CURVATURE',
    'DEFAULT_GAMMA',
    'FORGETTING_THRESHOLD',
]
