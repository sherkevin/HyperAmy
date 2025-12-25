"""
链接构建模块

提供自动构建 chunk 之间链接关系的功能，基于双曲空间中的距离。
"""
import time
import logging
from typing import List, Dict, Optional
import torch

from .types import Point
from .physics import ParticleProjector

logger = logging.getLogger(__name__)


def _poincare_dist(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    计算两个双曲坐标之间的庞加莱距离
    
    Args:
        u, v: 庞加莱球内的坐标向量
        
    Returns:
        双曲距离（非负浮点数）
    """
    sq_dist = torch.sum((u - v) ** 2)
    u_sq = torch.sum(u ** 2)
    v_sq = torch.sum(v ** 2)
    denom = (1 - u_sq) * (1 - v_sq) + 1e-7
    arg = 1 + 2 * sq_dist / denom
    # 边界保护：确保 arg >= 1，避免 acosh 域错误
    arg = torch.clamp(arg, min=1.0 + 1e-7)
    return torch.acosh(arg).item()


def build_hyperbolic_links(
    points: List[Point],
    projector: ParticleProjector,
    distance_threshold: float = 1.5,
    top_k: Optional[int] = None,
    t_now: Optional[float] = None
) -> Dict[str, List[str]]:
    """
    基于庞加莱球中的双曲距离自动构建链接关系
    
    邻域定义：在双曲空间中距离小于阈值的点之间建立链接。
    
    Args:
        points: Point 对象列表
        projector: 投影器实例，用于计算双曲坐标
        distance_threshold: 双曲距离阈值，默认 1.5
                           - 在庞加莱球中，距离范围通常是 [0, inf)
                           - 0.5-1.0: 非常近的点（相似情感和强度）
                           - 1.0-2.0: 中等距离（相似情感但强度不同）
                           - 2.0+: 较远的点（不同情感）
        top_k: 每个点的最大邻居数，None 表示不限制
        t_now: 当前时间戳，None 表示使用当前系统时间
    
    Returns:
        字典，key 是 point.id，value 是邻居 ID 列表（按距离从小到大排序）
    """
    if t_now is None:
        t_now = time.time()
    
    if not points:
        return {}
    
    logger.info(f"Building hyperbolic links for {len(points)} points "
                f"(threshold={distance_threshold}, top_k={top_k})")
    
    # 预计算所有点的双曲坐标
    point_states = {}
    for point in points:
        state = projector.compute_state(
            vec=point.emotion_vector,
            v=point.v,
            T=point.T,
            born=point.born,
            t_now=t_now
        )
        point_states[point.id] = state
    
    # 构建邻域关系
    edges = {}
    total_edges = 0
    
    for i, point in enumerate(points):
        neighbors_with_dist = []
        
        for j, other_point in enumerate(points):
            if point.id == other_point.id:
                continue
            
            # 计算双曲距离
            dist = _poincare_dist(
                point_states[point.id]['current_vector'],
                point_states[other_point.id]['current_vector']
            )
            
            # 如果距离小于阈值，添加到邻居列表
            if dist < distance_threshold:
                neighbors_with_dist.append((other_point.id, dist))
        
        # 按距离排序
        neighbors_with_dist.sort(key=lambda x: x[1])
        
        # 应用 top_k 限制
        if top_k is not None and len(neighbors_with_dist) > top_k:
            neighbors_with_dist = neighbors_with_dist[:top_k]
        
        # 提取邻居 ID
        neighbors = [nid for nid, _ in neighbors_with_dist]
        edges[point.id] = neighbors
        total_edges += len(neighbors)
        
        if (i + 1) % 100 == 0:
            logger.debug(f"Processed {i + 1}/{len(points)} points")
    
    logger.info(f"Built {total_edges} links (average {total_edges/len(points):.2f} per point)")
    return edges


def update_points_with_links(
    points: List[Point],
    edges: Dict[str, List[str]]
) -> List[Point]:
    """
    使用构建的链接关系更新 Point 对象
    
    Args:
        points: 原始 Point 对象列表
        edges: 链接关系字典（从 build_hyperbolic_links 返回）
    
    Returns:
        更新后的 Point 对象列表
    """
    # 创建 ID 到 Point 的映射
    point_dict = {point.id: point for point in points}
    
    # 更新每个点的 links
    updated_points = []
    for point in points:
        if point.id in edges:
            # 创建新的 Point 对象，更新 links
            updated_point = Point(
                id=point.id,
                emotion_vector=point.emotion_vector,
                v=point.v,
                T=point.T,
                born=point.born,
                current_time=point.current_time,
                links=edges[point.id]
            )
            updated_points.append(updated_point)
        else:
            # 如果没有链接，保持原样
            updated_points.append(point)
    
    return updated_points


def auto_link_points(
    points: List[Point],
    projector: ParticleProjector,
    distance_threshold: float = 1.5,
    top_k: Optional[int] = None,
    t_now: Optional[float] = None,
    update_inplace: bool = False
) -> List[Point]:
    """
    自动为 Point 列表构建链接关系的便捷函数
    
    Args:
        points: Point 对象列表
        projector: 投影器实例
        distance_threshold: 双曲距离阈值，默认 1.5
        top_k: 每个点的最大邻居数，None 表示不限制
        t_now: 当前时间戳，None 表示使用当前系统时间
        update_inplace: 是否原地更新（当前实现总是返回新对象）
    
    Returns:
        更新了 links 的 Point 对象列表
    """
    edges = build_hyperbolic_links(
        points=points,
        projector=projector,
        distance_threshold=distance_threshold,
        top_k=top_k,
        t_now=t_now
    )
    
    return update_points_with_links(points, edges)

