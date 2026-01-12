"""
存储层模块 (H-Mem System V3)

负责粒子对象的持久化存储，支持 V3 粒子系统。

存储内容：
- direction: 语义方向 μ（归一化的 emotion_vector）
- mass: 引力质量 m
- temperature: 热力学温度 T
- initial_radius: 初始双曲半径 R₀
- created_at: 创建时间 t₀
"""
import logging
import time
import json
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Union

import numpy as np

from ods import ChromaClient
from poincare.retrieval import CandidateParticle

if TYPE_CHECKING:
    # 仅用于类型检查，避免运行时导入
    from particle import ParticleEntity

logger = logging.getLogger(__name__)


class HyperAmyStorage:
    """
    存储层：负责粒子对象的持久化（调用层）

    支持 V3 粒子系统（H-Mem）和旧系统的 ParticleEntity。
    """
    def __init__(
        self,
        persist_path: Optional[str] = None,
        collection_name: str = "emotion_particles",
        curvature: float = 1.0
    ):
        """
        Args:
            persist_path: 数据库持久化路径。如果为 None，则根据 collection_name 自动生成。
            collection_name: 集合名称
            curvature: 空间曲率 c（用于计算初始半径）
        """
        # 如果未提供 persist_path，则根据 collection_name 隐式生成
        if persist_path is None:
            persist_path = f"./hyperamy_db_{collection_name}"

        self.ods_client = ChromaClient(persist_path=persist_path, collection_name=collection_name)
        self.curvature = curvature
        logger.info(f"HyperAmyStorage initialized: path={persist_path}, collection={collection_name}")

    def upsert_entity(self, entity: 'ParticleEntity', links: Optional[List[str]] = None):
        """
        存储或更新粒子实体
        
        Args:
            entity: 要存储的 ParticleEntity 对象
            links: 邻居实体 ID 列表（可选）
            
        Raises:
            Exception: 存储失败时抛出异常
        """
        try:
            # 1. 向量归一化：使用 ods 层的工具方法
            embedding_list = ChromaClient.normalize_vector(entity.emotion_vector)

            # 2. 准备 Metadata
            metadata = {
                "v": float(entity.speed),
                "T": float(entity.temperature),
                "weight": float(entity.weight),  # 粒子质量
                "born": float(entity.born),
                "last_updated": float(time.time()),
                "text_id": entity.text_id,
                "entity": entity.entity,
                "links": json.dumps(links or []),  # 序列化为 JSON 字符串
                # 新增：支持时间演化
                "purity": float(entity.purity),
                "tau_v": float(entity.tau_v),
                "tau_T": float(entity.tau_T)
            }
            
            # 序列化元数据（处理列表和字典）
            serialized_metadata = ChromaClient.serialize_metadata(metadata)

            # 3. 调用 ods 层写入
            self.ods_client.upsert(
                ids=[entity.entity_id],
                embeddings=[embedding_list],
                metadatas=[serialized_metadata]
            )
            
            logger.debug(f"Upserted entity: {entity.entity_id}")
        except Exception as e:
            logger.error(f"Failed to upsert entity {entity.entity_id}: {str(e)}")
            raise e
    
    def upsert_entities(self, entities: List['ParticleEntity'], links_map: Optional[Dict[str, List[str]]] = None):
        """
        批量存储或更新粒子实体
        
        Args:
            entities: ParticleEntity 对象列表
            links_map: 实体 ID 到邻居列表的映射（可选）
            
        Raises:
            Exception: 存储失败时抛出异常
        """
        if not entities:
            return
        
        try:
            ids = []
            embeddings = []
            metadatas = []
            
            for entity in entities:
                # 向量归一化
                embedding_list = ChromaClient.normalize_vector(entity.emotion_vector)
                embeddings.append(embedding_list)
                
                # 准备 Metadata
                links = (links_map or {}).get(entity.entity_id, [])

                # 提取标准 entity_id（如果粒子 ID 格式为 text_id_standard_entity_id）
                # 例如：full_test-abc_entity-count123 → entity-count123
                standard_entity_id = entity.entity_id
                if '_' in entity.entity_id and entity.entity_id.startswith(entity.text_id):
                    # 粒子 ID 包含 text_id 前缀，提取标准的 entity_id 部分
                    standard_entity_id = entity.entity_id[len(entity.text_id) + 1:]

                metadata = {
                    "v": float(entity.speed),
                    "T": float(entity.temperature),
                    "weight": float(entity.weight),  # 粒子质量
                    "born": float(entity.born),
                    "last_updated": float(time.time()),
                    "text_id": entity.text_id,
                    "conversation_id": entity.text_id,  # text_id 就是 conversation_id（在 Amygdala 工作流中）
                    "entity": entity.entity,
                    "entity_id": standard_entity_id,  # 标准 entity_id（用于与 HippoRAG 匹配）
                    "links": json.dumps(links),
                    # 新增：支持时间演化
                    "purity": float(entity.purity),
                    "tau_v": float(entity.tau_v),
                    "tau_T": float(entity.tau_T)
                }
                metadatas.append(ChromaClient.serialize_metadata(metadata))
                
                ids.append(entity.entity_id)
            
            # 批量写入
            self.ods_client.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Batch upserted {len(entities)} entities")
        except Exception as e:
            logger.error(f"Failed to batch upsert entities: {str(e)}")
            raise e
    
    @property
    def collection(self):
        """
        访问底层的 ChromaDB collection（用于检索层）

        注意：此属性仅用于向后兼容检索层的实现。
        新代码应该通过 ods_client 访问。
        """
        return self.ods_client.collection

    # ========== V3 粒子系统方法 ==========

    def upsert_candidate(self, candidate: CandidateParticle, direction_raw: Optional[np.ndarray] = None):
        """
        存储或更新 V3 候选粒子

        Args:
            candidate: 候选粒子对象
            direction_raw: 原始（未归一化）的方向向量，可选。
                          如果提供，将用于存储原始语义信息。

        Raises:
            Exception: 存储失败时抛出异常
        """
        try:
            # 方向向量（归一化）- 直接传递 numpy 数组
            embedding_list = ChromaClient.normalize_vector(candidate.direction)

            # 准备 V3 元数据
            metadata = {
                # V3 核心属性
                "mass": float(candidate.mass),
                "temperature": float(candidate.temperature),
                "initial_radius": float(candidate.initial_radius),
                "created_at": float(candidate.created_at),
                "last_updated": float(time.time()),
                # 存储原始维度（用于重建向量）
                "direction_dim": len(candidate.direction),
                # 用户自定义元数据
                **{f"meta_{k}": str(v) for k, v in candidate.metadata.items()}
            }

            # 如果提供了原始方向向量，存储它
            if direction_raw is not None:
                metadata["direction_raw"] = json.dumps(direction_raw.tolist())

            # 序列化元数据
            serialized_metadata = ChromaClient.serialize_metadata(metadata)

            # 调用 ods 层写入
            self.ods_client.upsert(
                ids=[candidate.id],
                embeddings=[embedding_list],
                metadatas=[serialized_metadata]
            )

            logger.debug(f"Upserted V3 candidate: {candidate.id}")
        except Exception as e:
            logger.error(f"Failed to upsert V3 candidate {candidate.id}: {str(e)}")
            raise e

    def upsert_candidates(self, candidates: List[CandidateParticle], direction_raws: Optional[List[np.ndarray]] = None):
        """
        批量存储或更新 V3 候选粒子

        Args:
            candidates: 候选粒子列表
            direction_raws: 原始方向向量列表（可选）

        Raises:
            Exception: 存储失败时抛出异常
        """
        if not candidates:
            return

        try:
            ids = []
            embeddings = []
            metadatas = []

            for i, candidate in enumerate(candidates):
                # 向量归一化 - 直接传递 numpy 数组
                embedding_list = ChromaClient.normalize_vector(candidate.direction)
                embeddings.append(embedding_list)

                # 准备元数据
                metadata = {
                    "mass": float(candidate.mass),
                    "temperature": float(candidate.temperature),
                    "initial_radius": float(candidate.initial_radius),
                    "created_at": float(candidate.created_at),
                    "last_updated": float(time.time()),
                    "direction_dim": len(candidate.direction),
                    **{f"meta_{k}": str(v) for k, v in candidate.metadata.items()}
                }

                # 如果提供了原始方向向量
                if direction_raws and i < len(direction_raws):
                    metadata["direction_raw"] = json.dumps(direction_raws[i].tolist())

                metadatas.append(ChromaClient.serialize_metadata(metadata))
                ids.append(candidate.id)

            # 批量写入
            self.ods_client.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"Batch upserted {len(candidates)} V3 candidates")
        except Exception as e:
            logger.error(f"Failed to batch upsert V3 candidates: {str(e)}")
            raise e

    def get_all_candidates(self, t_now: Optional[float] = None) -> List[CandidateParticle]:
        """
        获取所有存储的 V3 候选粒子

        Args:
            t_now: 当前时间（用于计算衰减状态）

        Returns:
            候选粒子列表
        """
        try:
            # 从数据库获取所有记录，明确请求 embeddings
            results = self.ods_client.collection.get(include=["embeddings", "metadatas"])

            # 检查结果是否有效
            if results is None:
                return []

            ids = results.get("ids")
            if ids is None or len(ids) == 0:
                return []

            candidates = []
            embeddings = results.get("embeddings")
            metadatas = results.get("metadatas")

            # 确保 embeddings 和 metadatas 是列表
            if embeddings is None:
                embeddings = []
            if metadatas is None:
                metadatas = []

            for i, particle_id in enumerate(ids):
                # 获取方向向量
                embedding = None
                if i < len(embeddings):
                    emb = embeddings[i]
                    if emb is not None:
                        # 检查是否是列表或数组类型
                        try:
                            direction = np.array(emb, dtype=np.float32)
                            norm = np.linalg.norm(direction)
                            if norm > 0:
                                direction = direction / norm
                                embedding = direction
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Failed to convert embedding for {particle_id}: {e}")

                if embedding is None:
                    # 没有嵌入向量，跳过
                    continue

                # 获取元数据
                metadata = {}
                if i < len(metadatas) and metadatas[i] is not None:
                    metadata = metadatas[i]

                # 提取 V3 属性
                mass = float(metadata.get("mass", 1.0))
                temperature = float(metadata.get("temperature", 1.0))
                initial_radius = float(metadata.get("initial_radius", 1.0))
                created_at = float(metadata.get("created_at", 0.0))

                # 提取用户元数据（过滤 meta_ 前缀的）
                user_metadata = {}
                for k, v in metadata.items():
                    if k.startswith("meta_"):
                        user_metadata[k[5:]] = v

                # 创建候选粒子
                candidate = CandidateParticle(
                    id=particle_id,
                    direction=embedding,
                    mass=mass,
                    temperature=temperature,
                    initial_radius=initial_radius,
                    created_at=created_at,
                    metadata=user_metadata
                )
                candidates.append(candidate)

            logger.debug(f"Retrieved {len(candidates)} V3 candidates")
            return candidates

        except Exception as e:
            logger.error(f"Failed to get V3 candidates: {str(e)}")
            return []

    def query_by_direction(
        self,
        direction: np.ndarray,
        n_results: int = 10,
        min_mass: Optional[float] = None,
        max_temp: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        按方向向量查询粒子（向量相似度搜索）

        Args:
            direction: 查询方向 μ_q
            n_results: 返回结果数量
            min_mass: 最小质量过滤（可选）
            max_temp: 最大温度过滤（可选）

        Returns:
            查询结果列表，每个结果包含 id, score, metadata 等
        """
        try:
            # 归一化查询向量并转换为列表（ChromaDB 期望列表格式）
            query_norm = np.linalg.norm(direction)
            if query_norm > 0:
                query_vector = (direction / query_norm).tolist()
            else:
                query_vector = direction.tolist()

            # 执行向量搜索
            results = self.ods_client.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )

            if not results or not results.get("ids"):
                return []

            formatted_results = []
            ids = results.get("ids", [])[0]
            distances = results.get("distances", [])[0]
            metadatas = results.get("metadatas", [])[0]

            for i, particle_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}

                # 应用过滤条件
                if min_mass is not None:
                    mass = float(metadata.get("mass", 1.0))
                    if mass < min_mass:
                        continue

                if max_temp is not None:
                    temp = float(metadata.get("temperature", 1.0))
                    if temp > max_temp:
                        continue

                formatted_results.append({
                    "id": particle_id,
                    "score": float(distances[i]) if i < len(distances) else 0.0,
                    "metadata": metadata
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to query by direction: {str(e)}")
            return []

    def delete_candidate(self, candidate_id: str) -> bool:
        """
        删除指定的候选粒子

        Args:
            candidate_id: 粒子 ID

        Returns:
            是否删除成功
        """
        try:
            self.ods_client.collection.delete(ids=[candidate_id])
            logger.debug(f"Deleted V3 candidate: {candidate_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete V3 candidate {candidate_id}: {str(e)}")
            return False

    def count_candidates(self) -> int:
        """
        获取存储的候选粒子数量

        Returns:
            粒子数量
        """
        try:
            return self.ods_client.collection.count()
        except Exception as e:
            logger.error(f"Failed to count candidates: {str(e)}")
            return 0

