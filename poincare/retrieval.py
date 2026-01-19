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
from poincare.physics import PhysicsEngine, ParticleState, compute_particle_state, DEFAULT_GAMMA as PHYSICS_DEFAULT_GAMMA
from poincare.types import SearchResult
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 默认参数
DEFAULT_SEMANTIC_THRESHOLD = 0.5  # 语义相似度阈值
DEFAULT_RETRIEVAL_BETA = 1.0       # 检索评分系数
DEFAULT_CURVATURE = 1.0            # 空间曲率
# ========== 方案二：显式使用physics.py中的DEFAULT_GAMMA ==========
# 确保使用physics.py中修改后的DEFAULT_GAMMA=0.001，而不是旧的1.0
DEFAULT_GAMMA = PHYSICS_DEFAULT_GAMMA  # 从physics模块导入最新的gamma值
# ========== 配置注入链修复结束 ==========


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
        # ========== 暴力验证模式：如果threshold < 0，则跳过Pruning ==========
        if self.config.semantic_threshold < 0:
            logger.warning(
                f"⚠️  暴力验证模式：Pruning已禁用（threshold={self.config.semantic_threshold}），"
                f"所有 {len(candidates)} 个候选粒子进入Thermodynamic Scoring阶段"
            )
            # 只做维度检查，不进行相似度过滤
            filtered = []
            for cand in candidates:
                if len(cand.direction) != len(query_direction):
                    logger.warning(
                        f"维度不匹配：候选向量维度={len(cand.direction)}，查询向量维度={len(query_direction)}，跳过该候选"
                    )
                    continue
                filtered.append(cand)
            return filtered
        # ========== 暴力验证模式结束 ==========
        
        filtered = []
        similarities = []  # 记录所有相似度用于调试

        for cand in candidates:
            # 维度检查：确保候选粒子向量与查询向量维度一致
            if len(cand.direction) != len(query_direction):
                logger.warning(
                    f"维度不匹配：候选向量维度={len(cand.direction)}，查询向量维度={len(query_direction)}，跳过该候选"
                )
                continue
            
            # 计算余弦相似度（即方向向量的点积，因为已归一化）
            try:
                similarity = float(np.dot(cand.direction, query_direction))
                similarities.append(similarity)
            except ValueError as e:
                logger.warning(f"计算相似度失败（维度不匹配？）：{e}，跳过该候选")
                continue

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
        # ========== 方案二：显式传递gamma值，确保使用0.001 ==========
        # ========== 暴力验证模式：将semantic_threshold设为-1.0以禁用Pruning ==========
        config = RetrievalConfig(
            semantic_threshold=-1.0,  # 暴力验证：设为-1.0禁用Pruning，让所有粒子进入Thermodynamic Scoring
            retrieval_beta=1.0,
            curvature=1.0,
            gamma=0.001  # 显式传递，确保使用修改后的gamma值
        )
        # ========== 暴力验证模式：Pruning已禁用 ==========
        # ========== 显式传递gamma修复结束 ==========
        self._retrieval = InMemoryRetrieval(config=config)
        
        # 保存physics引擎引用，供search_hybrid使用
        self.physics = self._retrieval.physics
        self.config = self._retrieval.config

        # 记录存储粒子的最早born时间，用于相对时间基准
        # 初始化为None，会在_load_particles_from_storage中设置
        self._storage_base_time = None

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
            
            # 第一遍：收集所有born时间戳，找到最早的有效时间作为基准
            born_times = []
            load_time = time.time()  # 记录加载时间，用于替换无效的born时间
            
            for i in range(len(ids)):
                if i >= len(metadatas) or metadatas[i] is None:
                    continue
                meta = metadatas[i]
                born_raw = meta.get("born", None)
                
                # 处理born时间戳：如果是None、0、负数或无效值，使用加载时间
                if born_raw is None or born_raw == 0 or born_raw == "0" or float(born_raw) <= 0:
                    born = load_time  # 使用加载时间
                    logger.debug(f"粒子 {ids[i]} 的born时间戳无效({born_raw})，使用加载时间: {born}")
                else:
                    try:
                        born = float(born_raw)
                        # 检查born时间戳是否合理（不能是未来的时间，也不能太早）
                        if born > load_time:
                            born = load_time  # 如果是未来时间，使用加载时间
                            logger.debug(f"粒子 {ids[i]} 的born时间戳是未来时间，使用加载时间")
                        elif born < load_time - 86400 * 365:  # 如果超过1年，可能无效
                            born = load_time  # 使用加载时间
                            logger.debug(f"粒子 {ids[i]} 的born时间戳太早，使用加载时间")
                        else:
                            born_times.append(born)
                    except (ValueError, TypeError):
                        born = load_time  # 转换失败，使用加载时间
                        logger.debug(f"粒子 {ids[i]} 的born时间戳转换失败，使用加载时间")
            
            # 设置存储基准时间：使用最早的born时间，如果没有有效的born时间，使用加载时间
            if born_times:
                self._storage_base_time = min(born_times)
                logger.info(f"存储基准时间已设置: {self._storage_base_time} (最早粒子时间, {len(born_times)}个有效粒子)")
            else:
                # 如果没有有效的born时间戳，使用加载时间作为基准
                self._storage_base_time = load_time
                logger.warning(f"未找到有效的born时间戳，使用加载时间作为基准: {load_time}")
            
            # 第二遍：加载粒子，将born时间调整为相对于基准时间的偏移
            for i, pid in enumerate(ids):
                if i >= len(embeddings) or embeddings[i] is None:
                    continue

                direction = np.array(embeddings[i], dtype=np.float32)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm

                meta = metadatas[i] if i < len(metadatas) else {}
                mass_raw = float(meta.get("weight", 1.0))
                # 设置最小质量，避免质量太小导致快速衰减
                mass = max(mass_raw, 0.5)  # 最小质量为0.5
                temperature = float(meta.get("T", 1.0))
                
                # 处理born时间戳：如果无效，使用基准时间
                born_raw = meta.get("born", None)
                if born_raw is None or born_raw == 0 or born_raw == "0" or float(born_raw) <= 0:
                    # 如果born时间戳无效，使用基准时间（偏移为0）
                    born = 0.0
                else:
                    try:
                        born_absolute = float(born_raw)
                        # 将born时间调整为相对于基准时间的偏移（秒），最小为0
                        born = max(0.0, born_absolute - self._storage_base_time)
                    except (ValueError, TypeError):
                        # 转换失败，使用0（相对于基准时间）
                        born = 0.0

                # 计算初始半径（设置最小值避免小质量粒子立即被遗忘）
                from poincare.physics import PhysicsEngine, DEFAULT_GAMMA
                physics = PhysicsEngine(curvature=1.0, gamma=DEFAULT_GAMMA)  # 使用较小的gamma减缓衰减
                initial_radius = max(2.0 * mass, 0.5)  # 最小初始半径为0.5，避免快速衰减

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
            self._storage_base_time = time.time()

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
        # 注意：t_now的设置逻辑在后面，这里不提前设置

        # 从 query_entity 提取信息
        query_direction = query_entity.emotion_vector
        
        # 确保query_direction是numpy数组
        if not isinstance(query_direction, np.ndarray):
            query_direction = np.array(query_direction, dtype=np.float32)
        else:
            query_direction = query_direction.astype(np.float32)
        
        # 维度验证：确保查询向量维度与存储的粒子向量维度一致
        if hasattr(self._retrieval, '_particles') and len(self._retrieval._particles) > 0:
            # 获取存储的粒子向量维度
            stored_dim = len(self._retrieval._particles[0].direction)
            query_dim = len(query_direction)
            
            if query_dim != stored_dim:
                logger.error(
                    f"维度不匹配！查询向量维度={query_dim}，存储向量维度={stored_dim}。"
                    f"请确保存储和检索使用相同的情绪提取器。"
                )
                # 返回空结果而不是崩溃
                return []
        
        norm = np.linalg.norm(query_direction)
        if norm > 0:
            query_direction = query_direction / norm
        else:
            logger.warning(f"查询向量全为0，返回空结果")
            return []

        query_mass_raw = getattr(query_entity, 'weight', 1.0)
        # 设置最小质量，避免质量太小导致快速衰减
        query_mass = max(query_mass_raw, 0.5)  # 最小质量为0.5
        query_temperature = getattr(query_entity, 'temperature', 1.0)
        query_initial_radius = max(2.0 * query_mass, 0.5)  # 最小初始半径为0.5

        # 使用相对时间基准：如果存储基准时间已设置，使用固定的小偏移作为t_now
        # 这样所有粒子的born时间都是相对于基准时间的偏移，delta_t会很小且可控
        if t_now is None:
            if hasattr(self, '_storage_base_time') and self._storage_base_time is not None:
                # 这是一个基于相对时间的系统，强制使用相对时间
                t_now = 300.0  # 固定使用300秒（5分钟）作为相对时间
                logger.debug(f"使用相对时间基准: t_now=300.0秒 (存储基准时间已应用)")
            else:
                # 这是一个实时流式系统（还没遇到过），使用挂钟时间
                t_now = time.time()
                logger.warning(f"未检测到存储基准时间，使用系统时间 t_now = {t_now}（可能导致delta_t过大）")
        
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

        # 设置最小质量，避免质量太小导致快速衰减
        mass = max(particle.weight, 0.5)  # 最小质量为0.5
        initial_radius = max(2.0 * mass, 0.5)  # 最小初始半径为0.5，避免快速衰减

        candidate = create_candidate(
            particle_id=particle.entity_id,
            direction=direction,
            mass=mass,
            temperature=particle.temperature,
            initial_radius=initial_radius,
            created_at=particle.born,
            conversation_id=getattr(particle, 'conversation_id', ''),
            entity=getattr(particle, 'entity', '')
        )
        self._retrieval.add_particle(candidate)

    def search_hybrid(
        self,
        query_text: str,
        query_entity,
        semantic_docs: List[str],
        semantic_scores: np.ndarray,
        id_to_content: Dict[str, str],
        top_k: int = 10,
        alpha: float = 0.8,  # 已弃用，现在使用动态权重
        t_now: Optional[float] = None
    ) -> List[SearchResult]:
        """
        全域平滑动态权重混合检索 (Continuous Dynamic Weighting Hybrid Search)
        
        根据Query情绪强度和语义置信度的博弈，动态决定情绪检索的权重，
        并使用Weighted RRF进行公平融合。
        
        Args:
            query_text: 查询文本
            query_entity: 查询粒子对象（ParticleEntity）
            semantic_docs: 语义检索得到的文档列表（Top-N，例如Top-100）
            semantic_scores: 语义分数（与semantic_docs对应）
            id_to_content: 映射（particle_id -> text_content）
            top_k: 最终返回结果数量
            alpha: 已弃用（保留以兼容旧代码）
            t_now: 当前时间
        
        Returns:
            重排序后的SearchResult列表
        """
        import math
        
        if not semantic_docs or len(semantic_docs) == 0:
            logger.warning("search_hybrid: 没有语义候选，返回空结果")
            return []
        
        # 构建content -> particle_id的逆映射
        content_to_id = {content: pid for pid, content in id_to_content.items()}
        
        # ========== 步骤A: 获取双路候选 (Dual Retrieval) ==========
        # 语义路：获取Top-50候选（从传入的Top-100中截取）
        semantic_candidates = []
        for doc, sem_score in zip(semantic_docs[:50], semantic_scores[:50]):
            particle_id = content_to_id.get(doc, None)
            if particle_id is None:
                continue
            candidate = self._retrieval._candidates.get(particle_id, None)
            if candidate is None:
                continue
            semantic_candidates.append((candidate, float(sem_score)))
        
        if not semantic_candidates:
            logger.warning(f"search_hybrid: 无法匹配语义候选到粒子，返回空结果")
            return []
        
        # 情绪路：获取Top-50候选（使用纯情绪检索）
        query_direction = query_entity.emotion_vector
        if not isinstance(query_direction, np.ndarray):
            query_direction = np.array(query_direction, dtype=np.float32)
        else:
            query_direction = query_direction.astype(np.float32)
        
        # 归一化查询向量
        norm = np.linalg.norm(query_direction)
        if norm > 0:
            query_direction = query_direction / norm
        
        query_mass = max(getattr(query_entity, 'weight', 1.0), 0.5)
        query_temperature = getattr(query_entity, 'temperature', 1.0)
        query_initial_radius = max(2.0 * query_mass, 0.5)
        
        # 设置时间
        if t_now is None:
            if hasattr(self, '_storage_base_time') and self._storage_base_time is not None:
                t_now = 300.0
            else:
                t_now = time.time()
        
        query_created_at = t_now
        
        # 计算查询状态
        query_state = self.physics.compute_state(
            direction=query_direction,
            mass=query_mass,
            temperature=query_temperature,
            initial_radius=query_initial_radius,
            created_at=query_created_at,
            t_now=t_now
        )
        
        # 对所有粒子计算情绪分数并排序，获取Top-50
        all_candidates = list(self._retrieval._candidates.values())
        emotion_scores_all = []
        for candidate in all_candidates:
            try:
                candidate_state = self.physics.compute_state(
                    direction=candidate.direction,
                    mass=candidate.mass,
                    temperature=candidate.temperature,
                    initial_radius=candidate.initial_radius,
                    created_at=candidate.created_at,
                    t_now=t_now
                )
                hyperbolic_dist = poincare_dist(
                    candidate_state.poincare_coord,
                    query_state.poincare_coord,
                    c=self.config.curvature
                )
                if hyperbolic_dist < 1e-9:
                    score = 1e6
                else:
                    temp_modulation = 1.0 + self.config.retrieval_beta / candidate_state.temperature
                    score = 1.0 / (hyperbolic_dist * temp_modulation)
                emotion_scores_all.append((candidate, score))
            except Exception as e:
                logger.debug(f"search_hybrid: 计算情绪分数失败（particle_id={candidate.id}）: {e}")
                emotion_scores_all.append((candidate, 0.0))
        
        # 按情绪分数降序排序，获取Top-50
        emotion_candidates = sorted(emotion_scores_all, key=lambda x: x[1], reverse=True)[:50]
        
        # ========== 步骤B: 计算关键指标 ==========
        # I_q (Query Emotion Intensity): Query情绪向量的L2范数
        query_emotion_raw = query_entity.emotion_vector
        if isinstance(query_emotion_raw, np.ndarray):
            I_q = float(np.linalg.norm(query_emotion_raw))
        else:
            I_q = float(np.linalg.norm(np.array(query_emotion_raw)))
        
        # 归一化到[0.0, 1.0]范围（如果向量未归一化）
        # 通常情绪向量已经归一化，但这里取最大值作为强度指标更稳健
        if isinstance(query_emotion_raw, np.ndarray):
            I_q = float(np.max(np.abs(query_emotion_raw)))
        else:
            I_q = float(np.max(np.abs(np.array(query_emotion_raw))))
        I_q = min(1.0, max(0.0, I_q))  # 确保在[0.0, 1.0]范围内
        
        # S_sem (Semantic Confidence): Top-1的语义分数
        if semantic_candidates:
            top1_score = semantic_candidates[0][1]
            # 确保分数在[0.0, 1.0]范围内（如果是余弦相似度通常已经是；如果是距离需转换）
            # 假设传入的分数已经是归一化的相似度分数
            S_sem = min(1.0, max(0.0, float(top1_score)))
        else:
            S_sem = 0.0
        
        # ========== 步骤C: 全域平滑动态定权 + 语义崩溃协议 ==========
        # 超参数
        k = 10.0       # Sigmoid 陡峭度
        bias = 0.15    # 语义主场优势
        min_sem_weight = 0.7  # 最低语义保护权重（仅在语义未崩溃时生效）
        SEMANTIC_COLLAPSE_THRESHOLD = 0.05  # 语义崩溃阈值：S_sem < 0.05 视为彻底失效
        
        # ========== 语义崩溃协议 (Semantic Collapse Protocol) ==========
        if S_sem < SEMANTIC_COLLAPSE_THRESHOLD:
            # 语义崩溃，解除安全锁！
            # 权重完全由情绪强度决定，最高可达 0.95（仍保留5%语义作为兜底）
            logger.warning(
                f"⚠️ Semantic Collapse Detected (S_sem={S_sem:.4f} < {SEMANTIC_COLLAPSE_THRESHOLD})! "
                f"Releasing Safety Lock (I_q={I_q:.4f})."
            )
            # 情绪权重：基础0.5 + 根据I_q调整（最高0.95）
            w_emo = 0.5 + (I_q * 0.45)  # I_q=1.0时，w_emo=0.95
            w_sem = 1.0 - w_emo  # 剩余权重给语义（最低5%兜底）
        else:
            # ========== 正常情况：保持原有的保护逻辑 ==========
            # 基础博弈：Sigmoid函数
            delta = I_q - (S_sem + bias)
            base_weight = 1.0 / (1.0 + math.exp(-k * delta))
            
            # 连续语义抑制
            suppression = 1.0 - (S_sem ** 2)
            
            # 计算原始情绪权重
            w_emo_raw = base_weight * suppression
            
            # 最低语义保护：确保语义权重不低于min_sem_weight
            # 这样即使语义置信度很低，也不会完全依赖情绪检索（因为情绪检索在QA任务中无效）
            w_sem_protected = max(min_sem_weight, 1.0 - w_emo_raw)
            w_emo = 1.0 - w_sem_protected
            w_sem = w_sem_protected
        
        # 确保权重在[0.0, 1.0]范围内
        w_emo = min(1.0, max(0.0, w_emo))
        w_sem = min(1.0, max(0.0, w_sem))
        
        # ========== 步骤E: 详细日志 ==========
        # 记录是否触发了语义崩溃协议
        if S_sem < SEMANTIC_COLLAPSE_THRESHOLD:
            logger.info(
                f"Dynamic Weighting: Iq={I_q:.4f}, S_sem={S_sem:.4f} -> "
                f"[COLLAPSE PROTOCOL] -> "
                f"Final W_emo={w_emo:.4f}, W_sem={w_sem:.4f}"
            )
        else:
            logger.info(
                f"Dynamic Weighting: Iq={I_q:.4f}, S_sem={S_sem:.4f} -> "
                f"Base={base_weight:.4f}, Supp={suppression:.4f} -> "
                f"Final W_emo={w_emo:.4f}, W_sem={w_sem:.4f}"
            )
        
        # ========== 步骤D: 加权RRF融合 (Weighted RRF) ==========
        final_scores = {}
        
        # 处理语义结果
        for rank, (candidate, sem_score) in enumerate(semantic_candidates, start=1):
            particle_id = candidate.id
            if particle_id not in final_scores:
                final_scores[particle_id] = {
                    'particle': candidate,
                    'semantic_score': sem_score,
                    'emotion_score': 0.0,
                    'semantic_rank': rank,
                    'emotion_rank': None
                }
            final_scores[particle_id]['score'] = final_scores[particle_id].get('score', 0.0) + w_sem / (60.0 + rank)
        
        # 处理情绪结果
        for rank, (candidate, emo_score) in enumerate(emotion_candidates, start=1):
            particle_id = candidate.id
            if particle_id not in final_scores:
                final_scores[particle_id] = {
                    'particle': candidate,
                    'semantic_score': 0.0,
                    'emotion_score': emo_score,
                    'semantic_rank': None,
                    'emotion_rank': rank
                }
            else:
                final_scores[particle_id]['emotion_score'] = emo_score
                final_scores[particle_id]['emotion_rank'] = rank
            final_scores[particle_id]['score'] = final_scores[particle_id].get('score', 0.0) + w_emo / (60.0 + rank)
        
        # 排序输出：按加权RRF分数降序排列
        sorted_results = sorted(
            final_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_k]
        
        # 构建返回结果
        results = []
        for particle_id, data in sorted_results:
            candidate = data['particle']
            
            # 计算双曲距离（用于SearchResult）
            candidate_state = self.physics.compute_state(
                direction=candidate.direction,
                mass=candidate.mass,
                temperature=candidate.temperature,
                initial_radius=candidate.initial_radius,
                created_at=candidate.created_at,
                t_now=t_now
            )
            hyperbolic_dist = poincare_dist(
                candidate_state.poincare_coord,
                query_state.poincare_coord,
                c=self.config.curvature
            )
            
            results.append(SearchResult(
                id=particle_id,
                score=data['score'],  # 使用加权RRF分数
                hyperbolic_distance=hyperbolic_dist,
                poincare_coord=candidate_state.poincare_coord,
                metadata={
                    **candidate.metadata,
                    'semantic_score': float(data['semantic_score']),
                    'emotion_score': float(data['emotion_score']),
                    'semantic_rank': data['semantic_rank'],
                    'emotion_rank': data['emotion_rank'],
                    'w_emo': float(w_emo),
                    'w_sem': float(w_sem),
                    'I_q': float(I_q),
                    'S_sem': float(S_sem),
                    'fusion_method': 'Weighted_RRF_Dynamic'  # 标记使用加权RRF动态融合
                }
            ))
        
        logger.info(
            f"search_hybrid: 完成动态权重混合检索（语义候选={len(semantic_candidates)}, "
            f"情绪候选={len(emotion_candidates)}, W_emo={w_emo:.4f}, W_sem={w_sem:.4f}），"
            f"返回Top-{top_k}结果"
        )
        
        return results


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
