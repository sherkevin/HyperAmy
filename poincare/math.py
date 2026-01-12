"""
Poincaré Ball Math Module (H-Mem System V3)

庞加莱球基础数学运算，实现双曲几何的核心函数。

根据 system_v3.md 设计文档：
- 几何舞台：庞加莱球 (𝔻_c^d, g^𝔻)，曲率为 -c
- 保角因子：λ_z = 2 / (1 - c||z||²)
- 双曲距离：d_𝔻(u, v) = (2/√c) * arctanh(√c * ||-u ⊕_c v||)

Author: HyperAmy Team
Version: 3.0
"""

import math
import numpy as np
import torch
from typing import Union, Optional

# 数值稳定性常量
EPS = 1e-8


def conformal_factor(z: Union[np.ndarray, torch.Tensor], c: float = 1.0) -> float:
    """
    计算保角因子 λ_z

    公式: λ_z = 2 / (1 - c * ||z||²)

    Args:
        z: 庞加莱球中的坐标向量
        c: 曲率参数（默认 1.0）

    Returns:
        保角因子值

    Raises:
        ValueError: 如果向量超出球体边界 (||z|| >= 1/√c)
    """
    if isinstance(z, torch.Tensor):
        norm_sq = torch.sum(z ** 2).item()
    else:
        norm_sq = np.sum(z ** 2)

    # 检查边界条件
    max_norm_sq = 1.0 / c - EPS
    if norm_sq >= max_norm_sq:
        raise ValueError(
            f"Vector norm squared {norm_sq:.6f} exceeds boundary {max_norm_sq:.6f}"
        )

    return 2.0 / (1.0 - c * norm_sq)


def mobius_add(
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor],
    c: float = 1.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Möbius 加法：u ⊕_c v

    公式:
        u ⊕_c v = ((1 + 2c<u,v> + c||v||²)u + (1 - c||u||²)v) / (1 + 2c<u,v> + c²||u||²||v||²)

    这是庞加莱球中的"向量加法"，对应于沿测地线的移动。

    Args:
        u, v: 庞加莱球中的坐标向量
        c: 曲率参数（默认 1.0）

    Returns:
        Möbius 加法结果
    """
    # 统一转换为 torch.Tensor 进行计算
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u).float()
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v).float()

    # 计算各项
    u_norm_sq = torch.sum(u ** 2)
    v_norm_sq = torch.sum(v ** 2)
    dot_uv = torch.sum(u * v)

    # 公式分子和分母
    numerator_u = (1.0 + 2.0 * c * dot_uv + c * v_norm_sq) * u
    numerator_v = (1.0 - c * u_norm_sq) * v
    numerator = numerator_u + numerator_v

    denominator = 1.0 + 2.0 * c * dot_uv + (c ** 2) * u_norm_sq * v_norm_sq

    result = numerator / (denominator + EPS)

    return result


def poincare_dist(
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor],
    c: float = 1.0,
    eps: float = EPS
) -> float:
    """
    计算庞加莱球中的双曲距离

    公式: d_𝔻(u, v) = (2/√c) * arctanh(√c * ||-u ⊕_c v||)

    这是连接两点 u 和 v 的测地线长度。

    Args:
        u, v: 庞加莱球中的坐标向量
        c: 曲率参数（默认 1.0）
        eps: 数值稳定性常数

    Returns:
        双曲距离（非负浮点数）
    """
    sqrt_c = math.sqrt(c)

    # 计算 -u ⊕_c v（从 u 指向 v 的"向量"）
    neg_u = -u if isinstance(u, torch.Tensor) else -u
    diff = mobius_add(neg_u, v, c)

    # 计算模长
    if isinstance(diff, torch.Tensor):
        diff_norm = torch.norm(diff).item()
    else:
        diff_norm = np.linalg.norm(diff)

    # 双曲距离公式
    arg = sqrt_c * diff_norm

    # 数值稳定性：arctanh(x) 在 x→1 时发散
    arg = min(arg, 1.0 - eps)

    distance = (2.0 / sqrt_c) * math.atanh(arg)

    return distance


def poincare_dist_batch(
    u: Union[np.ndarray, torch.Tensor],
    v_batch: Union[np.ndarray, torch.Tensor],
    c: float = 1.0
) -> np.ndarray:
    """
    批量计算双曲距离（u 到多个 v）

    优化版本，用于检索场景。

    Args:
        u: 查询向量
        v_batch: 候选向量矩阵，shape (n, dim)
        c: 曲率参数

    Returns:
        距离数组，shape (n,)
    """
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    if isinstance(v_batch, torch.Tensor):
        v_batch = v_batch.detach().cpu().numpy()

    n = v_batch.shape[0]
    distances = np.zeros(n)

    for i in range(n):
        distances[i] = poincare_dist(u, v_batch[i], c)

    return distances


def project_to_poincare(
    direction: Union[np.ndarray, torch.Tensor],
    radius: float,
    c: float = 1.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    将方向和双曲半径投影到庞加莱球坐标

    公式: z = tanh(√c * R / 2) / √c * μ

    Args:
        direction: 单位方向向量 μ
        radius: 双曲半径 R
        c: 曲率参数

    Returns:
        庞加莱球坐标向量 z
    """
    # 统一类型
    if isinstance(direction, np.ndarray):
        direction = torch.from_numpy(direction).float()

    # 归一化方向
    norm = torch.norm(direction)
    if norm < EPS:
        return torch.zeros_like(direction)
    unit_dir = direction / norm

    # 计算欧氏半径
    sqrt_c = math.sqrt(c)
    euclidean_radius = math.tanh(sqrt_c * radius / 2.0) / sqrt_c

    # 庞加莱坐标
    z = euclidean_radius * unit_dir

    return z


def extract_radius(
    z: Union[np.ndarray, torch.Tensor],
    c: float = 1.0
) -> float:
    """
    从庞加莱坐标提取双曲半径

    公式: R = (2/√c) * arctanh(√c * ||z||)

    Args:
        z: 庞加莱球坐标
        c: 曲率参数

    Returns:
        双曲半径 R
    """
    if isinstance(z, torch.Tensor):
        z_norm = torch.norm(z).item()
    else:
        z_norm = np.linalg.norm(z)

    sqrt_c = math.sqrt(c)

    # 反向双曲投影
    arg = sqrt_c * z_norm
    arg = min(arg, 1.0 - EPS)  # 数值稳定性

    radius = (2.0 / sqrt_c) * math.atanh(arg)

    return radius


def extract_direction(
    z: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    从庞加莱坐标提取单位方向向量

    Args:
        z: 庞加莱球坐标

    Returns:
        单位方向向量 μ
    """
    if isinstance(z, torch.Tensor):
        norm = torch.norm(z)
        if norm < EPS:
            return torch.zeros_like(z)
        return z / norm
    else:
        norm = np.linalg.norm(z)
        if norm < EPS:
            return np.zeros_like(z)
        return z / norm


class PoincareBall:
    """
    庞加莱球空间类

    封装双曲几何运算，提供统一的接口。
    """

    def __init__(self, curvature: float = 1.0, dimension: Optional[int] = None):
        """
        初始化庞加莱球空间

        Args:
            curvature: 曲率 c（默认 1.0）
            dimension: 空间维度（可选）
        """
        self.c = curvature
        self.sqrt_c = math.sqrt(curvature)
        self.dimension = dimension

    def project(self, direction: np.ndarray, radius: float) -> np.ndarray:
        """投影到庞加莱球"""
        result = project_to_poincare(direction, radius, self.c)
        return result.detach().cpu().numpy() if isinstance(result, torch.Tensor) else result

    def dist(self, u: np.ndarray, v: np.ndarray) -> float:
        """计算双曲距离"""
        return poincare_dist(u, v, self.c)

    def mobius(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Möbius 加法"""
        result = mobius_add(u, v, self.c)
        return result.detach().cpu().numpy() if isinstance(result, torch.Tensor) else result

    def get_radius(self, z: np.ndarray) -> float:
        """提取双曲半径"""
        return extract_radius(z, self.c)

    def get_direction(self, z: np.ndarray) -> np.ndarray:
        """提取方向向量"""
        result = extract_direction(z)
        return result.detach().cpu().numpy() if isinstance(result, torch.Tensor) else result

    def __repr__(self) -> str:
        return f"PoincareBall(curvature={self.c}, dimension={self.dimension})"


# 导出函数和类
__all__ = [
    'conformal_factor',
    'mobius_add',
    'poincare_dist',
    'poincare_dist_batch',
    'project_to_poincare',
    'extract_radius',
    'extract_direction',
    'PoincareBall',
]
