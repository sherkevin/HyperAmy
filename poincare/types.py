"""
数据类型定义模块

定义 HyperAmy 系统中的核心数据结构：
- SearchResult: 检索结果数据结构

注意：Point 类已移除，poincare 模块直接使用 particle.ParticleEntity。
ParticleEntity 从 particle 模块导入，使用延迟导入避免循环依赖。
"""
from dataclasses import dataclass
from typing import Dict, Any, Literal, List, TYPE_CHECKING

if TYPE_CHECKING:
    # 仅用于类型检查，避免运行时导入
    from particle import ParticleEntity


@dataclass
class SearchResult:
    """
    检索结果数据结构，提供明确的返回格式
    """
    id: str
    score: float          # 双曲距离 (越小越好)
    metadata: Dict[str, Any]
    vector: List[float]
    match_type: Literal['direct', 'neighbor']  # 匹配类型：直接检索或邻域扩展

