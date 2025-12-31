"""
链接构建模块

提供自动构建 chunk 之间链接关系的功能，基于双曲空间中的距离。
"""
import time
import logging
from typing import List, Dict, Optional, TYPE_CHECKING
import torch

from .physics import ParticleProjector

if TYPE_CHECKING:
    # 仅用于类型检查，避免运行时导入
    from particle import ParticleEntity

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
    entities: List['ParticleEntity'],
    projector: ParticleProjector,
    distance_threshold: float = 1.5,
    top_k: Optional[int] = None,
    t_now: Optional[float] = None
) -> Dict[str, List[str]]:
    """
    基于庞加莱球中的双曲距离自动构建链接关系
    
    邻域定义：在双曲空间中距离小于阈值的点之间建立链接。
    
    Args:
        entities: ParticleEntity 对象列表
        projector: 投影器实例，用于计算双曲坐标
        distance_threshold: 双曲距离阈值，默认 1.5
                           - 在庞加莱球中，距离范围通常是 [0, inf)
                           - 0.5-1.0: 非常近的点（相似情感和强度）
                           - 1.0-2.0: 中等距离（相似情感但强度不同）
                           - 2.0+: 较远的点（不同情感）
        top_k: 每个点的最大邻居数，None 表示不限制
        t_now: 当前时间戳，None 表示使用当前系统时间
    
    Returns:
        字典，key 是 entity.entity_id，value 是邻居 ID 列表（按距离从小到大排序）
    """
    if t_now is None:
        t_now = time.time()
    
    if not entities:
        return {}
    
    logger.info(f"Building hyperbolic links for {len(entities)} entities "
                f"(threshold={distance_threshold}, top_k={top_k})")
    
    # 预计算所有实体的双曲坐标
    entity_states = {}
    for entity in entities:
        # ParticleEntity.emotion_vector 是归一化后的方向向量，compute_state 会自动转换
        state = projector.compute_state(
            vec=entity.emotion_vector,
            v=entity.speed,
            T=entity.temperature,
            born=entity.born,
            t_now=t_now,
            weight=entity.weight
        )
        # 跳过已消失的粒子
        if state.get('is_expired', False):
            continue
        entity_states[entity.entity_id] = state
    
    # 构建邻域关系
    edges = {}
    total_edges = 0
    
    for i, entity in enumerate(entities):
        neighbors_with_dist = []
        
        for j, other_entity in enumerate(entities):
            if entity.entity_id == other_entity.entity_id:
                continue
            
            # 计算双曲距离
            dist = _poincare_dist(
                entity_states[entity.entity_id]['current_vector'],
                entity_states[other_entity.entity_id]['current_vector']
            )
            
            # 如果距离小于阈值，添加到邻居列表
            if dist < distance_threshold:
                neighbors_with_dist.append((other_entity.entity_id, dist))
        
        # 按距离排序
        neighbors_with_dist.sort(key=lambda x: x[1])
        
        # 应用 top_k 限制
        if top_k is not None and len(neighbors_with_dist) > top_k:
            neighbors_with_dist = neighbors_with_dist[:top_k]
        
        # 提取邻居 ID
        neighbors = [nid for nid, _ in neighbors_with_dist]
        edges[entity.entity_id] = neighbors
        total_edges += len(neighbors)
        
        if (i + 1) % 100 == 0:
            logger.debug(f"Processed {i + 1}/{len(entities)} entities")
    
    logger.info(f"Built {total_edges} links (average {total_edges/len(entities):.2f} per entity)")
    return edges


def update_entities_with_links(
    entities: List['ParticleEntity'],
    edges: Dict[str, List[str]]
) -> List['ParticleEntity']:
    """
    使用构建的链接关系更新 ParticleEntity 对象
    
    注意：ParticleEntity 是 dataclass，但 links 不是其字段。
    此函数返回链接映射字典，实际更新应在存储层进行。
    
    Args:
        entities: 原始 ParticleEntity 对象列表
        edges: 链接关系字典（从 build_hyperbolic_links 返回）
    
    Returns:
        原始 entities 列表（ParticleEntity 本身不包含 links 字段）
    """
    # ParticleEntity 本身不包含 links 字段，links 存储在数据库中
    # 此函数主要用于兼容性，实际返回原列表
    # 调用者应该使用 edges 字典来更新数据库中的 links
    logger.debug(f"Links map prepared for {len(entities)} entities")
    return entities


def auto_link_entities(
    entities: List['ParticleEntity'],
    projector: ParticleProjector,
    distance_threshold: float = 1.5,
    top_k: Optional[int] = None,
    t_now: Optional[float] = None
) -> Dict[str, List[str]]:
    """
    自动为 ParticleEntity 列表构建链接关系的便捷函数
    
    Args:
        entities: ParticleEntity 对象列表
        projector: 投影器实例
        distance_threshold: 双曲距离阈值，默认 1.5
        top_k: 每个点的最大邻居数，None 表示不限制
        t_now: 当前时间戳，None 表示使用当前系统时间
    
    Returns:
        链接关系字典，key 是 entity_id，value 是邻居 ID 列表
    """
    edges = build_hyperbolic_links(
        entities=entities,
        projector=projector,
        distance_threshold=distance_threshold,
        top_k=top_k,
        t_now=t_now
    )
    
    return edges

