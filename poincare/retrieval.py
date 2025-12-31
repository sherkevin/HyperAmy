"""
检索层模块

实现混合检索流水线：锥体 -> 壳层 -> 邻域 -> 汇总
"""
import time
import json
import logging
import torch
import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

from .types import SearchResult
from .storage import HyperAmyStorage
from .physics import ParticleProjector
from ods import ChromaClient

if TYPE_CHECKING:
    # 仅用于类型检查，避免运行时导入
    from particle import ParticleEntity

logger = logging.getLogger(__name__)


class HyperAmyRetrieval:
    """
    检索层：实现混合检索流水线
    
    四步检索流程：
    1. 锥体锁定：使用向量相似度快速圈定方向一致的粒子
    2. 壳层筛选：计算真实的双曲距离进行精排
    3. 邻域激活：从 Top-K 点出发，扩展其邻居节点
    4. 汇总排序：混合直接检索点和邻居点，最终排序返回
    """
    def __init__(self, storage: HyperAmyStorage, projector: ParticleProjector):
        """
        Args:
            storage: 存储层实例
            projector: 投影器实例
        """
        self.storage = storage
        self.projector = projector

    def _poincare_dist(self, u: torch.Tensor, v: torch.Tensor) -> float:
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
        # 当坐标接近球面边界时，可能出现数值不稳定
        arg = torch.clamp(arg, min=1.0 + 1e-7)
        return torch.acosh(arg).item()

    def _calculate_score_raw(self, 
                           dynamic_query: dict, 
                           cand_vec: List[float], 
                           cand_meta: Dict[str, Any], 
                           t_now: float) -> float:
        """
        计算查询点与候选点的双曲距离（性能优化版本）
        
        直接使用原始数值，避免构建 Point 对象的开销。
        适用于批量计算场景。
        
        Args:
            dynamic_query: 查询点的动态状态（已预计算）
            cand_vec: 候选点的向量（归一化后的方向向量）
            cand_meta: 候选点的元数据
            t_now: 当前时间戳
            
        Returns:
            双曲距离（如果粒子已消失，返回 inf）
        """
        # 计算候选点的动态状态
        # cand_vec 是 List[float]，需要转换为 numpy array
        cand_vec_array = np.array(cand_vec)
        weight = cand_meta.get('weight', 1.0)  # 获取 weight，默认为 1.0（向后兼容）
        dynamic_cand = self.projector.compute_state(
            vec=cand_vec_array,
            v=cand_meta['v'],
            T=cand_meta['T'],
            born=cand_meta['born'],
            t_now=t_now,
            weight=weight
        )
        
        # 如果候选粒子已消失，返回无穷大距离
        if dynamic_cand.get('is_expired', False):
            return float('inf')
        
        # 计算双曲距离
        return self._poincare_dist(
            dynamic_query['current_vector'],
            dynamic_cand['current_vector']
        )

    def search(self, 
               query_entity: 'ParticleEntity', 
               top_k: int = 3, 
               cone_width: int = 50, 
               max_neighbors: int = 20,
               neighbor_penalty: float = 1.1) -> List[SearchResult]:
        """
        执行混合检索
        
        Args:
            query_entity: 查询粒子实体
            top_k: 返回结果数量
            cone_width: 锥体搜索宽度。建议值 50-100。
                       值越大，召回率越高，但后续物理计算开销越大。
                       应根据总数据量 N 动态调整 (如 logN * 10)。
            max_neighbors: 邻域扩展时的最大节点数，防止超级节点导致性能雪崩。
            neighbor_penalty: 邻居惩罚系数，默认 1.1 (邻居距离会被放大 10%)。
                             用于降低间接连接的优先级。
        
        Returns:
            检索结果列表，按双曲距离从小到大排序
        """
        t_now = time.time()
        
        # 预计算 Query 状态（避免在循环中重复计算）
        # ParticleEntity.emotion_vector 是归一化后的方向向量，compute_state 会自动转换
        dynamic_query = self.projector.compute_state(
            vec=query_entity.emotion_vector,
            v=query_entity.speed,
            T=query_entity.temperature,
            born=query_entity.born,
            t_now=t_now,
            weight=query_entity.weight
        )
        
        # 如果查询粒子已消失，返回空结果
        if dynamic_query.get('is_expired', False):
            logger.warning(f"Query particle {query_entity.entity_id} has expired (distance >= max_radius)")
            return []

        # --- Step 1: Cone Search (锥体锁定) ---
        # 使用 ods 层的归一化方法
        query_vec_list = ChromaClient.normalize_vector(query_entity.emotion_vector)
        
        try:
            # 通过 storage 的 ods_client 进行查询
            results = self.storage.ods_client.query(
                query_embeddings=[query_vec_list],
                n_results=cone_width,
                include=["metadatas", "embeddings"]
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

        if not results['ids'] or not results['ids'][0]:
            return []

        # --- Step 2: Distance Ranking (壳层/距离筛选) ---
        ids = results['ids'][0]
        metas = results['metadatas'][0]
        vecs = results['embeddings'][0]

        scored_candidates = []
        
        for pid, meta, vec in zip(ids, metas, vecs):
            try:
                score = self._calculate_score_raw(dynamic_query, vec, meta, t_now)
                # 跳过已消失的粒子（距离为 inf）
                if score == float('inf'):
                    logger.debug(f"Skipping expired particle {pid}")
                    continue
                scored_candidates.append(SearchResult(
                    id=pid, 
                    score=score, 
                    metadata=meta, 
                    vector=vec, 
                    match_type='direct'
                ))
            except Exception as e:
                logger.warning(f"Error calculating score for {pid}: {e}")
                continue

        # 初步排序，取前 K 个进入邻域扩展
        scored_candidates.sort(key=lambda x: x.score)
        top_candidates = scored_candidates[:top_k]

        # --- Step 3: Neighborhood Expansion (邻域激活) ---
        neighbor_ids = []
        for cand in top_candidates:
            try:
                links_str = cand.metadata.get('links', '[]')
                links = json.loads(links_str)
                # 性能保护：限制单个节点的扩展数量
                if len(links) > max_neighbors:
                    links = links[:max_neighbors]
                neighbor_ids.extend(links)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse links for {cand.id}: {e}")
        
        # 去重并过滤已存在的点
        existing_ids = {c.id for c in top_candidates}
        neighbor_ids = list(set(neighbor_ids) - existing_ids)
        
        # 全局性能保护：限制总扩展数量
        if len(neighbor_ids) > max_neighbors * top_k:
            neighbor_ids = neighbor_ids[:max_neighbors * top_k]

        if neighbor_ids:
            try:
                # 通过 storage 的 ods_client 获取邻居节点
                nb_results = self.storage.ods_client.get(
                    ids=neighbor_ids, 
                    include=["metadatas", "embeddings"]
                )
                if nb_results['ids']:
                    for i, nid in enumerate(nb_results['ids']):
                        meta = nb_results['metadatas'][i]
                        vec = nb_results['embeddings'][i]
                        try:
                            score = self._calculate_score_raw(dynamic_query, vec, meta, t_now)
                            # 跳过已消失的粒子（距离为 inf）
                            if score == float('inf'):
                                logger.debug(f"Skipping expired neighbor particle {nid}")
                                continue
                            # 应用邻居惩罚系数
                            score *= neighbor_penalty 
                            
                            top_candidates.append(SearchResult(
                                id=nid, 
                                score=score, 
                                metadata=meta, 
                                vector=vec, 
                                match_type='neighbor'
                            ))
                        except Exception as e:
                            logger.debug(f"Error calculating neighbor score for {nid}: {e}")
                            continue
            except Exception as e:
                logger.error(f"Neighbor fetch failed: {e}")

        # --- Step 4: Final Aggregation (汇总与最终排序) ---
        top_candidates.sort(key=lambda x: x.score)
        return top_candidates[:top_k]

