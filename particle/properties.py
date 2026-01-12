"""
Particle Properties Module (H-Mem System V3)

粒子属性计算模块，实现解耦表征：
- Semantic Direction (μ): 语义方向
- Gravitational Mass (m): 引力质量
- Thermodynamic Temperature (T): 热力学温度

根据 system_v3.md 设计文档：
- m = α·I + β·log(1 + κ)
- T = T₀/κ
- R₀ ∝ m

其中：
- I = 情绪强度 (emotion vector 的模长)
- κ = 分布纯度 (L2/L1 norm ratio)

Author: HyperAmy Team
Version: 3.0
"""

import math
import numpy as np
from typing import Dict, Optional, Union
from dataclasses import dataclass

from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 默认参数
DEFAULT_ALPHA_MASS = 1.0       # 质量公式系数 α
DEFAULT_BETA_MASS = 1.0        # 质量公式系数 β
DEFAULT_T0 = 1.0               # 基准温度
DEFAULT_RADIUS_SCALE = 1.0     # 半径缩放系数 (R₀ = radius_scale * m)
MIN_KAPPA = 1e-6               # 最小纯度（避免除零）


@dataclass
class ParticleProperties:
    """
    粒子属性数据类

    Attributes:
        direction: 语义方向向量 μ（归一化）
        intensity: 情绪强度 I
        purity: 分布纯度 κ (0, 1]
        mass: 引力质量 m
        temperature: 热力学温度 T
        initial_radius: 初始双曲半径 R₀
    """
    direction: np.ndarray      # 语义方向 μ (归一化)
    intensity: float           # 情绪强度 I
    purity: float              # 分布纯度 κ
    mass: float                # 引力质量 m
    temperature: float         # 热力学温度 T
    initial_radius: float      # 初始双曲半径 R₀

    def __repr__(self) -> str:
        return (
            f"ParticleProperties("
            f"mass={self.mass:.4f}, "
            f"temperature={self.temperature:.4f}, "
            f"purity={self.purity:.4f}, "
            f"intensity={self.intensity:.4f}, "
            f"R0={self.initial_radius:.4f})"
        )


class PropertyCalculator:
    """
    粒子属性计算器

    实现 system_v3.md 中的解耦表征计算。
    """

    def __init__(
        self,
        alpha_mass: float = DEFAULT_ALPHA_MASS,
        beta_mass: float = DEFAULT_BETA_MASS,
        T0: float = DEFAULT_T0,
        radius_scale: float = DEFAULT_RADIUS_SCALE,
        min_kappa: float = MIN_KAPPA
    ):
        """
        初始化属性计算器

        Args:
            alpha_mass: 质量公式系数 α (强度权重)
            beta_mass: 质量公式系数 β (纯度权重)
            T0: 基准温度
            radius_scale: 半径缩放系数
            min_kappa: 最小纯度（避免除零）
        """
        self.alpha_mass = alpha_mass
        self.beta_mass = beta_mass
        self.T0 = T0
        self.radius_scale = radius_scale
        self.min_kappa = min_kappa

        logger.info(
            f"PropertyCalculator initialized: "
            f"α={alpha_mass}, β={beta_mass}, T0={T0}, "
            f"radius_scale={radius_scale}"
        )

    def compute_intensity(self, emotion_vector: np.ndarray) -> float:
        """
        计算情绪强度 I

        I = ||e|| (emotion vector 的模长)

        Args:
            emotion_vector: 情绪向量（未归一化）

        Returns:
            情绪强度 I
        """
        intensity = float(np.linalg.norm(emotion_vector))
        return max(intensity, 0.0)

    def compute_purity(self, emotion_vector: np.ndarray) -> float:
        """
        计算分布纯度 κ

        κ = ||e||²₂ / ||e||²₁ = L2² / L1²

        纯度范围：[1/d, 1]，其中 d 是向量维度
        - κ → 1: 所有能量集中在单个维度（高纯度）
        - κ → 1/d: 能量均匀分布（低纯度）

        Args:
            emotion_vector: 情绪向量（未归一化）

        Returns:
            分布纯度 κ ∈ (0, 1]
        """
        l2_norm = np.linalg.norm(emotion_vector)
        l1_norm = np.linalg.norm(emotion_vector, ord=1)

        if l1_norm < self.min_kappa:
            return self.min_kappa

        # 纯度 = L2² / L1²
        purity = (l2_norm ** 2) / (l1_norm ** 2)

        # 限制在有效范围内
        d = len(emotion_vector)
        min_purity = 1.0 / d
        purity = max(purity, min_purity)

        return float(purity)

    def compute_mass(self, intensity: float, purity: float) -> float:
        """
        计算引力质量 m

        m = α·I + β·log(1 + κ)

        物理意义：
        - 质量越大，抗拒遗忘的能力越强（惯性大）
        - 强度 I 线性贡献
        - 纯度 κ 对数贡献（边际递减）

        Args:
            intensity: 情绪强度 I
            purity: 分布纯度 κ

        Returns:
            引力质量 m
        """
        mass = (
            self.alpha_mass * intensity +
            self.beta_mass * math.log(1.0 + purity)
        )
        return max(mass, 0.01)  # 避免质量为零

    def compute_temperature(self, purity: float) -> float:
        """
        计算热力学温度 T

        T = T₀ / κ

        物理意义：
        - 温度越高，粒子的热运动范围（检索半径）越大
        - 高纯度 → 低温（精确命中）
        - 低纯度 → 高温（模糊匹配）

        Args:
            purity: 分布纯度 κ

        Returns:
            热力学温度 T
        """
        # 避免除零
        kappa_safe = max(purity, self.min_kappa)
        temperature = self.T0 / kappa_safe

        # 设置温度上限（防止纯度极低时温度爆炸）
        max_temp = self.T0 / self.min_kappa
        temperature = min(temperature, max_temp)

        return float(temperature)

    def compute_initial_radius(self, mass: float) -> float:
        """
        计算初始双曲半径 R₀

        R₀ ∝ m (质量越大，离黑洞越远，越清晰)

        Args:
            mass: 引力质量 m

        Returns:
            初始双曲半径 R₀
        """
        return self.radius_scale * mass

    def compute_properties(
        self,
        emotion_vector: np.ndarray
    ) -> ParticleProperties:
        """
        计算粒子的所有属性

        Args:
            emotion_vector: 情绪向量（未归一化）

        Returns:
            ParticleProperties 对象
        """
        # 1. 计算情绪强度 I
        intensity = self.compute_intensity(emotion_vector)

        # 2. 计算分布纯度 κ
        purity = self.compute_purity(emotion_vector)

        # 3. 计算语义方向 μ（归一化）
        norm = np.linalg.norm(emotion_vector)
        if norm < self.min_kappa:
            direction = np.zeros_like(emotion_vector)
        else:
            direction = emotion_vector / norm

        # 4. 计算引力质量 m
        mass = self.compute_mass(intensity, purity)

        # 5. 计算热力学温度 T
        temperature = self.compute_temperature(purity)

        # 6. 计算初始双曲半径 R₀
        initial_radius = self.compute_initial_radius(mass)

        return ParticleProperties(
            direction=direction,
            intensity=intensity,
            purity=purity,
            mass=mass,
            temperature=temperature,
            initial_radius=initial_radius
        )

    def compute_batch(
        self,
        emotion_vectors: list,
        entity_ids: Optional[list] = None
    ) -> Dict[str, list]:
        """
        批量计算粒子属性

        Args:
            emotion_vectors: 情绪向量列表
            entity_ids: 实体 ID 列表（可选，用于日志）

        Returns:
            属性字典，包含 direction, intensity, purity, mass, temperature, initial_radius
        """
        n = len(emotion_vectors)
        directions = []
        intensities = []
        purities = []
        masses = []
        temperatures = []
        initial_radii = []

        for i, vec in enumerate(emotion_vectors):
            props = self.compute_properties(vec)

            directions.append(props.direction)
            intensities.append(props.intensity)
            purities.append(props.purity)
            masses.append(props.mass)
            temperatures.append(props.temperature)
            initial_radii.append(props.initial_radius)

            if entity_ids:
                logger.debug(
                    f"Entity {entity_ids[i]}: "
                    f"I={props.intensity:.4f}, κ={props.purity:.4f}, "
                    f"m={props.mass:.4f}, T={props.temperature:.4f}, "
                    f"R₀={props.initial_radius:.4f}"
                )

        logger.info(f"Computed properties for {n} particles")

        return {
            'directions': directions,
            'intensities': intensities,
            'purities': purities,
            'masses': masses,
            'temperatures': temperatures,
            'initial_radii': initial_radii
        }


# 默认计算器实例
_default_calculator: Optional[PropertyCalculator] = None


def get_default_calculator() -> PropertyCalculator:
    """获取默认计算器实例"""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = PropertyCalculator()
    return _default_calculator


def compute_properties(emotion_vector: np.ndarray) -> ParticleProperties:
    """
    便捷函数：使用默认计算器计算粒子属性

    Args:
        emotion_vector: 情绪向量

    Returns:
        ParticleProperties 对象
    """
    calculator = get_default_calculator()
    return calculator.compute_properties(emotion_vector)


# 导出
__all__ = [
    'ParticleProperties',
    'PropertyCalculator',
    'get_default_calculator',
    'compute_properties',
    'DEFAULT_ALPHA_MASS',
    'DEFAULT_BETA_MASS',
    'DEFAULT_T0',
    'DEFAULT_RADIUS_SCALE',
    'MIN_KAPPA',
]
