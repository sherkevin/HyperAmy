"""
Retrieval Module (H-Mem System V3)

检索层：实现三步检索流程

根据 system_v3.md 设计文档：
1. 锥体语义过滤: cos(μ_i, μ_q) > η
2. 引力投影: O(1) 位置更新
3. 热力学采样: 温度调制的双曲距离评分

评分公式: Score = 1 / (d_hyp * (1 + β/T))

Author: HyperAmy Team
Version: 3.0
"""
import time
import math
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

import numpy as np

from poincare.math import poincare_dist, PoincareBall
from poincare.physics import PhysicsEngine, ParticleState, compute_particle_state
from poincare.types import SearchResult
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 默认参数
DEFAULT_SEMANTIC_THRESHOLD = 0.5  # 语义相似度阈值
DEFAULT_RETRIEVAL_BETA = 1.0       # 检索评分系数
DEFAULT_CURVATURE = 1.0            # 空间曲率
DEFAULT_GAMMA = 1.0                # 衰变常数


@dataclass
class RetrievalConfig:
    """检索配置"""
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD  # η: 语义相似度阈值
    retrieval_beta: float = DEFAULT_RETRIEVAL_BETA          # β: 温度调制系数
    curvature: float = DEFAULT_CURVATURE                    # c: 空间曲率
    gamma: float = DEFAULT_GAMMA                            # γ: 衰变常数
    forgetting_threshold: float = 1e-3                     # 遗忘阈值

    def __repr__(self) -> str:
        return (
            f"RetrievalConfig(η={self.semantic_threshold}, "
            f"β={self.retrieval_beta}, c={self.curvature}, γ={self.gamma})"
        )


@dataclass
class CandidateParticle:
    """
    候选粒子（用于检索中间状态）

    Attributes:
        id: 粒子 ID
        direction: 语义方向 μ
        mass: 引力质量 m
        temperature: 热力学温度 T
        initial_radius: 初始双曲半径 R₀
        created_at: 创建时间 t₀
        metadata: 原始元数据
    """
    id: str
    direction: np.ndarray
    mass: float
    temperature: float
    initial_radius: float
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    检索结果

    Attributes:
        id: 粒子 ID
        score: 检索分数
        hyperbolic_distance: 双曲距离
        semantic_similarity: 语义相似度
        temperature: 粒子温度
        memory_strength: 记忆强度
        metadata: 原始元数据
    """
    id: str
    score: float
    hyperbolic_distance: float
    semantic_similarity: float
    temperature: float
    memory_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"RetrievalResult(id={self.id[:20]}..., score={self.score:.4f})"


class HMemRetrieval:
    """
    H-Mem 检索系统 (System V3)

    实现三步检索流程：
    1. 锥体语义过滤 (Semantic Pruning)
    2. 引力投影 (Gravitational Projection)
    3. 热力学采样 (Thermodynamic Sampling)
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        physics_engine: Optional[PhysicsEngine] = None
    ):
        """
        初始化检索系统

        Args:
            config: 检索配置
            physics_engine: 物理引擎（可选）
        """
        self.config = config or RetrievalConfig()

        if physics_engine is None:
            self.physics = PhysicsEngine(
                curvature=self.config.curvature,
                gamma=self.config.gamma,
                forgetting_threshold=self.config.forgetting_threshold
            )
        else:
            self.physics = physics_engine

        logger.info(f"HMemRetrieval initialized with {self.config}")

    def _semantic_pruning(
        self,
        candidates: List[CandidateParticle],
        query_direction: np.ndarray
    ) -> List[CandidateParticle]:
        """
        Step 1: 锥体语义过滤

        过滤出与查询方向语义相似的候选粒子。

        公式: S_1 = { i | μ_i^T μ_q > η }

        Args:
            candidates: 候选粒子列表
            query_direction: 查询方向 μ_q

        Returns:
            过滤后的候选粒子列表
        """
        filtered = []
        similarities = []  # 记录所有相似度用于调试

        for cand in candidates:
            # 计算余弦相似度（即方向向量的点积，因为已归一化）
            similarity = float(np.dot(cand.direction, query_direction))
            similarities.append(similarity)

            # 只保留相似度高于阈值的
            if similarity >= self.config.semantic_threshold:
                filtered.append(cand)

        # 记录详细信息（INFO 级别，方便调试）
        if len(candidates) > 0:
            max_sim = max(similarities) if similarities else 0.0
            min_sim = min(similarities) if similarities else 0.0
            logger.info(
                f"Semantic pruning: {len(candidates)} -> {len(filtered)} "
                f"(threshold={self.config.semantic_threshold}, "
                f"max_sim={max_sim:.4f}, min_sim={min_sim:.4f})"
            )
        else:
            logger.warning("Semantic pruning: no candidates provided")

        return filtered

    def _gravitational_projection(
        self,
        candidates: List[CandidateParticle],
        t_now: float
    ) -> List[ParticleState]:
        """
        Step 2: 引力投影（O(1) 位置更新）

        对每个候选粒子，计算其在当前时刻的完整状态。

        R_i(t) = R_{0,i} · exp(-γ/m_i · Δt)
        z_i(t) = tanh(√c/2 · R_i(t)) · μ_i

        Args:
            candidates: 候选粒子列表
            t_now: 当前时间

        Returns:
            粒子状态列表
        """
        states = []

        for cand in candidates:
            state = self.physics.compute_state(
                direction=cand.direction,
                mass=cand.mass,
                temperature=cand.temperature,
                initial_radius=cand.initial_radius,
                created_at=cand.created_at,
                t_now=t_now
            )
            states.append(state)

        logger.debug(f"Gravitational projection: {len(candidates)} states computed")

        return states

    def _thermodynamic_scoring(
        self,
        candidate_states: List[ParticleState],
        query_state: ParticleState
    ) -> List[RetrievalResult]:
        """
        Step 3: 热力学采样（温度调制的距离评分）

        计算每个候选粒子的检索分数，分数越高越相关。

        公式: Score = 1 / (d_hyp(q, z_i) · (1 + β/T_i))

        物理意义：
        - d_hyp: 双曲距离，越小越相关
        - (1 + β/T): 温度调制因子
          - T 大（模糊记忆）→ 因子小 → 距离惩罚小 → 容易检索
          - T 小（清晰记忆）→ 因子大 → 距离惩罚大 → 需要精确匹配

        Args:
            candidate_states: 候选粒子状态列表
            query_state: 查询粒子状态

        Returns:
            检索结果列表，按分数降序排序
        """
        results = []
        forgotten_count = 0

        for state in candidate_states:
            # 跳过已遗忘的粒子
            if state.is_forgotten:
                forgotten_count += 1
                continue

            # 计算双曲距离
            hyp_dist = poincare_dist(
                query_state.poincare_coord,
                state.poincare_coord,
                c=self.config.curvature
            )

            # 计算语义相似度（用于显示）
            semantic_sim = float(np.dot(query_state.direction, state.direction))

            # 温度调制因子
            # T 越大 → 因子越小 → 距离惩罚越小
            temp_factor = 1.0 + self.config.retrieval_beta / state.temperature

            # 检索分数
            score = 1.0 / (hyp_dist * temp_factor + 1e-8)

            results.append(RetrievalResult(
                id=state.direction.tobytes()[:20].hex(),  # 临时 ID
                score=score,
                hyperbolic_distance=hyp_dist,
                semantic_similarity=semantic_sim,
                temperature=state.temperature,
                memory_strength=state.memory_strength,
                metadata={}
            ))

        # 记录详细信息
        logger.info(
            f"Thermodynamic scoring: {len(candidate_states)} input -> "
            f"{forgotten_count} forgotten -> {len(results)} results"
        )

        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def retrieve(
        self,
        query_direction: np.ndarray,
        query_mass: float,
        query_temperature: float,
        query_initial_radius: float,
        candidates: List[CandidateParticle],
        t_now: Optional[float] = None,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        执行完整的三步检索流程

        Args:
            query_direction: 查询方向 μ_q
            query_mass: 查询质量 m_q
            query_temperature: 查询温度 T_q
            query_initial_radius: 查询初始半径 R₀_q
            candidates: 候选粒子列表
            t_now: 当前时间（默认为系统时间）
            top_k: 返回结果数量

        Returns:
            检索结果列表，按分数降序排序
        """
        if t_now is None:
            t_now = time.time()

        # 计算查询状态
        query_created_at = t_now  # 假设查询是"现在"创建的

        # Step 1: 锥体语义过滤
        pruned = self._semantic_pruning(candidates, query_direction)

        if not pruned:
            return []

        # Step 2: 引力投影
        candidate_states = self._gravitational_projection(pruned, t_now)

        # 计算查询状态（用于距离计算）
        query_state = self.physics.compute_state(
            direction=query_direction,
            mass=query_mass,
            temperature=query_temperature,
            initial_radius=query_initial_radius,
            created_at=query_created_at,
            t_now=t_now
        )

        # Step 3: 热力学采样
        results = self._thermodynamic_scoring(candidate_states, query_state)

        # 返回 Top-K
        return results[:top_k]


class InMemoryRetrieval(HMemRetrieval):
    """
    内存检索系统（用于测试和原型）

    在内存中维护候选粒子列表，不依赖外部存储。
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None
    ):
        """
        初始化内存检索系统

        Args:
            config: 检索配置
        """
        super().__init__(config=config)
        self._candidates: Dict[str, CandidateParticle] = {}

    def add_particle(self, particle: CandidateParticle) -> None:
        """添加粒子到索引"""
        self._candidates[particle.id] = particle
        logger.debug(f"Added particle: {particle.id}")

    def add_particles(self, particles: List[CandidateParticle]) -> None:
        """批量添加粒子"""
        for p in particles:
            self._candidates[p.id] = p
        logger.info(f"Added {len(particles)} particles")

    def remove_particle(self, particle_id: str) -> bool:
        """移除粒子"""
        if particle_id in self._candidates:
            del self._candidates[particle_id]
            return True
        return False

    def get_all_candidates(self) -> List[CandidateParticle]:
        """获取所有候选粒子"""
        return list(self._candidates.values())

    def search(
        self,
        query_direction: np.ndarray,
        query_mass: float = 1.0,
        query_temperature: float = 1.0,
        query_initial_radius: float = 1.0,
        top_k: int = 10,
        t_now: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        在内存中执行检索

        Args:
            query_direction: 查询方向 μ_q
            query_mass: 查询质量
            query_temperature: 查询温度
            query_initial_radius: 查询初始半径
            top_k: 返回结果数量
            t_now: 当前时间

        Returns:
            检索结果列表
        """
        if t_now is None:
            t_now = time.time()

        candidates = list(self._candidates.values())

        # 执行检索，同时传递候选信息
        results = self._retrieve_with_ids(
            query_direction=query_direction,
            query_mass=query_mass,
            query_temperature=query_temperature,
            query_initial_radius=query_initial_radius,
            candidates=candidates,
            t_now=t_now,
            top_k=top_k
        )

        return results

    def _retrieve_with_ids(
        self,
        query_direction: np.ndarray,
        query_mass: float,
        query_temperature: float,
        query_initial_radius: float,
        candidates: List[CandidateParticle],
        t_now: float,
        top_k: int
    ) -> List[RetrievalResult]:
        """
        带ID的检索方法

        Args:
            query_direction: 查询方向
            query_mass: 查询质量
            query_temperature: 查询温度
            query_initial_radius: 查询初始半径
            candidates: 候选粒子列表
            t_now: 当前时间
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # Step 1: 锥体语义过滤
        pruned = self._semantic_pruning(candidates, query_direction)

        if not pruned:
            return []

        # Step 2: 引力投影
        candidate_states = self._gravitational_projection(pruned, t_now)

        # 计算查询状态
        query_state = self.physics.compute_state(
            direction=query_direction,
            mass=query_mass,
            temperature=query_temperature,
            initial_radius=query_initial_radius,
            created_at=t_now,
            t_now=t_now
        )

        # Step 3: 热力学采样（同时记录原始ID）
        results = []
        forgotten_count = 0

        for state, cand in zip(candidate_states, pruned):
            # 跳过已遗忘的粒子
            if state.is_forgotten:
                forgotten_count += 1
                continue

            # 计算双曲距离
            hyp_dist = poincare_dist(
                query_state.poincare_coord,
                state.poincare_coord,
                c=self.config.curvature
            )

            # 计算语义相似度
            semantic_sim = float(np.dot(query_state.direction, state.direction))

            # 温度调制因子
            temp_factor = 1.0 + self.config.retrieval_beta / state.temperature

            # 检索分数
            score = 1.0 / (hyp_dist * temp_factor + 1e-8)

            results.append(RetrievalResult(
                id=cand.id,  # 直接使用候选粒子的 ID
                score=score,
                hyperbolic_distance=hyp_dist,
                semantic_similarity=semantic_sim,
                temperature=state.temperature,
                memory_strength=state.memory_strength,
                metadata=cand.metadata
            ))

        # 记录详细信息
        logger.info(
            f"Thermodynamic scoring (with_ids): {len(candidate_states)} input -> "
            f"{forgotten_count} forgotten -> {len(results)} results"
        )

        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def __len__(self) -> int:
        """返回粒子数量"""
        return len(self._candidates)


# 便捷函数
def create_candidate(
    particle_id: str,
    direction: np.ndarray,
    mass: float,
    temperature: float,
    initial_radius: float,
    created_at: float,
    **metadata
) -> CandidateParticle:
    """
    创建候选粒子

    Args:
        particle_id: 粒子 ID
        direction: 语义方向 μ
        mass: 引力质量 m
        temperature: 热力学温度 T
        initial_radius: 初始双曲半径 R₀
        created_at: 创建时间
        **metadata: 额外的元数据

    Returns:
        CandidateParticle 对象
    """
    return CandidateParticle(
        id=particle_id,
        direction=direction,
        mass=mass,
        temperature=temperature,
        initial_radius=initial_radius,
        created_at=created_at,
        metadata=metadata
    )


# ========== 向后兼容适配器 ==========

class SearchResult:
    """
    向后兼容的搜索结果

    适配 V3 的 RetrievalResult 到旧系统的 SearchResult 接口。
    """
    def __init__(
        self,
        id: str,
        score: float,
        hyperbolic_distance: float,
        poincare_coord: np.ndarray,
        metadata: Dict[str, Any]
    ):
        self.id = id
        self.score = score  # 在旧系统中是距离（越小越好）
        self.hyperbolic_distance = hyperbolic_distance
        self.poincare_coord = poincare_coord
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id[:20]}..., score={self.score:.4f})"


class HyperAmyRetrieval:
    """
    向后兼容的检索类适配器

    适配旧系统的 HyperAmyRetrieval 接口，内部使用 V3 的 InMemoryRetrieval。

    旧接口参数：
    - storage: 粒子存储
    - projector: 粒子投影器

    V3 实现：
    - 使用 InMemoryRetrieval
    - 自动从存储加载粒子
    """

    def __init__(self, storage=None, projector=None):
        """
        初始化检索器（向后兼容）

        Args:
            storage: HyperAmyStorage 实例（可选）
            projector: ParticleProjector 实例（可选，V3 中不需要）
        """
        self.storage = storage
        self.projector = projector

        # 使用 V3 InMemoryRetrieval
        config = RetrievalConfig(
            semantic_threshold=0.5,
            retrieval_beta=1.0,
            curvature=1.0,
            gamma=1.0
        )
        self._retrieval = InMemoryRetrieval(config=config)

        # 如果提供了存储，自动加载粒子
        if storage is not None:
            self._load_particles_from_storage()

        logger.info("HyperAmyRetrieval initialized (V3 adapter)")

    def _load_particles_from_storage(self):
        """从存储加载粒子到内存检索器"""
        try:
            all_data = self.storage.collection.get(include=["embeddings", "metadatas"])
            ids = all_data.get("ids", [])
            embeddings = all_data.get("embeddings", [])
            metadatas = all_data.get("metadatas", [])

            for i, pid in enumerate(ids):
                if i >= len(embeddings) or embeddings[i] is None:
                    continue

                direction = np.array(embeddings[i], dtype=np.float32)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm

                meta = metadatas[i] if i < len(metadatas) else {}
                mass = float(meta.get("weight", 1.0))
                temperature = float(meta.get("T", 1.0))
                born = float(meta.get("born", time.time()))

                # 计算初始半径
                from poincare.physics import PhysicsEngine
                physics = PhysicsEngine(curvature=1.0, gamma=1.0)
                initial_radius = 2.0 * mass  # 使用 scaling_factor=2.0

                candidate = create_candidate(
                    particle_id=pid,
                    direction=direction,
                    mass=mass,
                    temperature=temperature,
                    initial_radius=initial_radius,
                    created_at=born,
                    conversation_id=meta.get("conversation_id", ""),
                    entity=meta.get("entity", "")
                )
                self._retrieval.add_particle(candidate)

            logger.debug(f"Loaded {len(ids)} particles from storage")

        except Exception as e:
            logger.warning(f"Failed to load particles from storage: {e}")

    def search(
        self,
        query_entity,
        top_k: int = 10,
        cone_width: float = 20.0,
        t_now: Optional[float] = None
    ) -> List[SearchResult]:
        """
        搜索相似的粒子（向后兼容接口）

        Args:
            query_entity: 查询粒子对象（ParticleEntity）
            top_k: 返回结果数量
            cone_width: 锥体宽度（V3 中转换为语义阈值）
            t_now: 当前时间

        Returns:
            SearchResult 列表
        """
        if t_now is None:
            t_now = time.time()

        # 从 query_entity 提取信息
        query_direction = query_entity.emotion_vector
        norm = np.linalg.norm(query_direction)
        if norm > 0:
            query_direction = query_direction / norm

        query_mass = getattr(query_entity, 'weight', 1.0)
        query_temperature = getattr(query_entity, 'temperature', 1.0)
        query_initial_radius = 2.0 * query_mass

        # 使用 V3 检索
        v3_results = self._retrieval.search(
            query_direction=query_direction,
            query_mass=query_mass,
            query_temperature=query_temperature,
            query_initial_radius=query_initial_radius,
            top_k=top_k,
            t_now=t_now
        )

        # 转换为向后兼容的 SearchResult 格式
        results = []
        for v3_result in v3_results:
            # 使用 hyperbolic_distance 作为 score（距离越小越好）
            results.append(SearchResult(
                id=v3_result.id,
                score=v3_result.hyperbolic_distance,
                hyperbolic_distance=v3_result.hyperbolic_distance,
                poincare_coord=np.array([]),  # 占位符
                metadata=v3_result.metadata
            ))

        return results

    def add_particle(self, particle):
        """添加粒子（向后兼容）"""
        direction = particle.emotion_vector
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        initial_radius = 2.0 * particle.weight

        candidate = create_candidate(
            particle_id=particle.entity_id,
            direction=direction,
            mass=particle.weight,
            temperature=particle.temperature,
            initial_radius=initial_radius,
            created_at=particle.born,
            conversation_id=getattr(particle, 'conversation_id', ''),
            entity=getattr(particle, 'entity', '')
        )
        self._retrieval.add_particle(candidate)


# 更新导出列表
__all__ = [
    'RetrievalConfig',
    'CandidateParticle',
    'RetrievalResult',
    'HMemRetrieval',
    'InMemoryRetrieval',
    'HyperAmyRetrieval',  # 向后兼容
    'SearchResult',        # 向后兼容
    'create_candidate',
    'DEFAULT_SEMANTIC_THRESHOLD',
    'DEFAULT_RETRIEVAL_BETA',
]
