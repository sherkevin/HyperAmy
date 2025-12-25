"""
数据类型定义模块

定义 HyperAmy 系统中的核心数据结构：
- Point: 基本粒子单位
- SearchResult: 检索结果数据结构
"""
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
import torch


@dataclass
class Point:
    """
    HyperAmy 系统中的基本粒子单位 (业务层对象)
    """
    id: str
    emotion_vector: torch.Tensor  # [Dim], 原始欧式空间方向向量
    v: float                      # 初始强度/速度 (v_init)
    T: float                      # 初始温度/熵 (T_init)
    born: float                   # 粒子生成时的绝对时间戳
    current_time: Optional[float] = None  # 数据记录更新的时间戳，None 表示使用当前时间
    links: List[str] = field(default_factory=list)  # 邻居粒子的ID列表

    def to_metadata(self) -> Dict[str, Any]:
        """
        序列化为 ChromaDB 支持的 Metadata 格式
        
        ChromaDB 的 metadata 只支持 int, float, str, bool 类型，
        列表和字典必须序列化为 JSON 字符串。
        """
        update_time = self.current_time if self.current_time is not None else time.time()
        return {
            "v": float(self.v),
            "T": float(self.T),
            "born": float(self.born),
            "last_updated": float(update_time),
            "links": json.dumps(self.links)  # 序列化为 JSON 字符串
        }


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

