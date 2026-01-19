"""
Amygdala 工作流类

整合粒子处理、存储和对话管理功能，提供统一的高级接口。
"""
import os
import time
import pandas as pd
import logging
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path

from particle import Particle, ParticleEntity
from poincare import HyperAmyStorage, ParticleProjector, auto_link_entities
from hipporag.embedding_store import EmbeddingStore
from hipporag.utils.misc_utils import compute_mdhash_id
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SimpleConversationStore:
    """
    简化的对话存储类
    
    当不需要嵌入向量时，只存储对话文本和 ID 映射关系。
    """
    
    def __init__(self, db_filename: str, namespace: str = "conversation"):
        """
        初始化简化的对话存储
        
        Args:
            db_filename: 存储目录路径
            namespace: 命名空间标识
        """
        self.namespace = namespace
        
        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)
        
        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        if os.path.exists(self.filename):
            try:
                df = pd.read_parquet(self.filename)
                # 兼容可能存在的 embedding 列（忽略）
                required_cols = ["hash_id", "content"]
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing required columns in {self.filename}, initializing empty store")
                    self.hash_ids, self.texts = [], []
                    self.hash_id_to_idx, self.hash_id_to_row = {}, {}
                    self.hash_id_to_text, self.text_to_hash_id = {}, {}
                    return
                
                self.hash_ids = df["hash_id"].values.tolist()
                self.texts = df["content"].values.tolist()
                self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
                self.hash_id_to_row = {
                    h: {"hash_id": h, "content": t}
                    for h, t in zip(self.hash_ids, self.texts)
                }
                self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
                self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
                logger.info(f"Loaded {len(self.hash_ids)} conversation records from {self.filename}")
            except Exception as e:
                logger.warning(f"Failed to load conversation store from {self.filename}: {e}, initializing empty store")
                self.hash_ids, self.texts = [], []
                self.hash_id_to_idx, self.hash_id_to_row = {}, {}
                self.hash_id_to_text, self.text_to_hash_id = {}, {}
        else:
            self.hash_ids, self.texts = [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}
            self.hash_id_to_text, self.text_to_hash_id = {}, {}
    
    def _save_data(self):
        """保存数据"""
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t}
            for h, t in zip(self.hash_ids, self.texts)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} conversation records to {self.filename}")
    
    def insert_strings(self, texts: List[str]):
        """
        插入对话文本
        
        Args:
            texts: 对话文本列表
        """
        logger.info(f"[SimpleConversationStore.insert_strings] 开始插入对话文本")
        logger.info(f"  输入文本数量: {len(texts)}")
        for i, text in enumerate(texts, 1):
            logger.info(f"  文本 {i}: {text[:100]}{'...' if len(text) > 100 else ''} (长度: {len(text)} 字符)")
        
        nodes_dict = {}
        
        for text in texts:
            hash_id = compute_mdhash_id(text, prefix=self.namespace + "-")
            nodes_dict[hash_id] = {'content': text}
            logger.debug(f"  文本 '{text[:50]}...' -> hash_id: {hash_id}")
        
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            logger.warning(f"[SimpleConversationStore.insert_strings] 没有有效的文本需要插入")
            return
        
        existing = set(self.hash_id_to_row.keys())
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        
        logger.info(
            f"[SimpleConversationStore.insert_strings] 插入统计: "
            f"{len(missing_ids)} 条新记录, "
            f"{len(all_hash_ids) - len(missing_ids)} 条已存在"
        )
        
        if not missing_ids:
            logger.info(f"[SimpleConversationStore.insert_strings] 所有文本已存在，跳过插入")
            return
        
        texts_to_insert = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        logger.info(f"[SimpleConversationStore.insert_strings] 准备插入 {len(missing_ids)} 条新记录")
        for i, (hash_id, text) in enumerate(zip(missing_ids, texts_to_insert), 1):
            logger.debug(f"  新记录 {i}: hash_id={hash_id}, text={text[:100]}{'...' if len(text) > 100 else ''}")
        
        # 使用空嵌入向量（占位）
        self.hash_ids.extend(missing_ids)
        self.texts.extend(texts_to_insert)
        
        logger.info(f"[SimpleConversationStore.insert_strings] 保存新记录到文件")
        self._save_data()
        logger.info(f"[SimpleConversationStore.insert_strings] 插入完成，当前总记录数: {len(self.hash_ids)}")
    
    def get_hash_id(self, text: str) -> Optional[str]:
        """根据文本获取 hash_id"""
        return self.text_to_hash_id.get(text)
    
    def get_text(self, hash_id: str) -> Optional[str]:
        """根据 hash_id 获取文本"""
        return self.hash_id_to_text.get(hash_id)
    
    def get_all_ids(self) -> List[str]:
        """获取所有 hash_id"""
        return self.hash_ids.copy()


class Amygdala:
    """
    Amygdala 工作流类
    
    整合粒子处理、存储和对话管理功能，提供统一的高级接口。
    
    主要功能：
    1. 处理对话文本，生成粒子实体
    2. 存储粒子到 ChromaDB
    3. 存储对话到 parquet 文件
    4. 维护粒子与对话的映射关系
    """
    
    def __init__(
        self,
        save_dir: str = "./workflow_data",
        particle_collection_name: str = "emotion_particles",
        conversation_namespace: str = "conversation",
        embedding_model: Optional[Any] = None,
        embedding_batch_size: int = 32,
        particle_model_name: Optional[str] = None,
        particle_embedding_model_name: Optional[str] = None,
        entity_extractor: Optional[Any] = None,
        sentence_processor: Optional[Any] = None,
        auto_link_particles: bool = True,
        link_distance_threshold: float = 1.5,
        link_top_k: Optional[int] = None
    ):
        """
        初始化 Amygdala 工作流
        
        Args:
            save_dir: 工作目录（存储对话和关系映射）
            particle_collection_name: 粒子存储的 collection 名称
            conversation_namespace: 对话存储的 namespace
            embedding_model: EmbeddingStore 需要的嵌入模型（可选）
            embedding_batch_size: 嵌入批处理大小
            particle_model_name: Particle 模块的 LLM 模型名称
            particle_embedding_model_name: Particle 模块的嵌入模型名称
            entity_extractor: 实体提取器（可选）
            sentence_processor: 句子处理器（可选）
            auto_link_particles: 是否自动构建粒子邻域链接
            link_distance_threshold: 邻域链接的距离阈值
            link_top_k: 每个粒子的最大邻域数量（None 表示不限制）
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.particle_collection_name = particle_collection_name
        self.conversation_namespace = conversation_namespace
        self.auto_link_particles = auto_link_particles
        self.link_distance_threshold = link_distance_threshold
        self.link_top_k = link_top_k
        
        # 初始化 Particle 模块
        self.particle = Particle(
            model_name=particle_model_name,
            embedding_model_name=particle_embedding_model_name,
            entity_extractor=entity_extractor,
            sentence_processor=sentence_processor
        )
        logger.info("Particle module initialized")
        
        # 初始化粒子存储
        self.particle_storage = HyperAmyStorage(
            collection_name=particle_collection_name
        )
        logger.info(f"Particle storage initialized: collection={particle_collection_name}")
        
        # 初始化粒子投影器（用于邻域链接）
        # max_radius 增大到 10000，允许粒子存在更长时间（约 5.5 小时）
        self.particle_projector = ParticleProjector(
            curvature=1.0,
            scaling_factor=2.0,
            max_radius=10000.0
        )
        
        # 初始化对话存储
        conversation_store_dir = str(self.save_dir / "conversation_embeddings")
        if embedding_model is not None:
            self.conversation_store = EmbeddingStore(
                embedding_model=embedding_model,
                db_filename=conversation_store_dir,
                batch_size=embedding_batch_size,
                namespace=conversation_namespace
            )
            logger.info("Conversation store initialized with embedding model")
        else:
            self.conversation_store = SimpleConversationStore(
                db_filename=conversation_store_dir,
                namespace=conversation_namespace
            )
            logger.info("Conversation store initialized without embedding model")
        
        # 初始化关系映射
        self.particle_to_conversation: Dict[str, str] = {}
        self.conversation_to_particles: Dict[str, List[str]] = {}
        self.relationship_mapping_file = self.save_dir / "particle_conversation_mapping.parquet"
        self._load_relationship_mapping()
        
        logger.info("Amygdala workflow initialized")
    
    def add(
        self,
        conversation: str,
        conversation_id: Optional[str] = None,
        entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        添加对话并处理

        流程：
        1. 生成或使用提供的 conversation_id
        2. 调用 Particle.process() 生成粒子列表
           - 如果提供了 entities，则使用预抽取的实体（避免重复LLM调用）
        3. 存储粒子到数据库
        4. 存储对话到数据库
        5. 记录粒子与对话的对应关系

        Args:
            conversation: 对话文本
            conversation_id: 对话 ID（可选，如果不提供则自动生成）
            entities: 预抽取的实体列表（可选，用于复用已抽取的结果）

        Returns:
            包含以下字段的字典：
            - conversation_id: 对话 ID
            - particles: 生成的粒子列表
            - particle_count: 粒子数量
            - relationship_map: particle_id -> conversation_id 映射
        """
        try:
            # Step 1: 生成或使用 conversation_id
            if conversation_id is None:
                conversation_id = compute_mdhash_id(
                    conversation,
                    prefix=f"{self.conversation_namespace}-"
                )

            # 记录输入信息
            logger.info("=" * 80)
            logger.info(f"[Amygdala.add] 开始处理对话")
            logger.info(f"  输入 - conversation_id: {conversation_id}")
            logger.info(f"  输入 - conversation_text: {conversation[:200]}{'...' if len(conversation) > 200 else ''}")
            logger.info(f"  输入 - conversation_length: {len(conversation)} 字符")
            logger.info(f"  输入 - entities: {entities if entities else 'None (将自动抽取)'}")

            # Step 2: 调用 Particle.process() 生成粒子列表
            # 如果提供了预抽取的实体，直接使用；否则自动抽取
            try:
                particles = self.particle.process(
                    text=conversation,
                    text_id=conversation_id,
                    entities=entities  # 传入预抽取的实体
                )
                logger.info(f"[Amygdala.add] 粒子生成完成: {len(particles)} 个粒子")
                
                # 详细记录每个粒子的信息
                if particles:
                    logger.info(f"[Amygdala.add] 粒子详细信息:")
                    for i, particle in enumerate(particles, 1):
                        emotion_vec_preview = particle.emotion_vector[:5].tolist() if len(particle.emotion_vector) >= 5 else particle.emotion_vector.tolist()
                        logger.info(
                            f"  粒子 {i}/{len(particles)}:"
                            f" entity_id={particle.entity_id},"
                            f" entity={particle.entity},"
                            f" speed={particle.speed:.6f},"
                            f" temperature={particle.temperature:.6f},"
                            f" weight={particle.weight:.6f},"
                            f" emotion_vector_shape={particle.emotion_vector.shape},"
                            f" emotion_vector_preview={emotion_vec_preview},"
                            f" born={particle.born}"
                        )
                else:
                    logger.warning(f"[Amygdala.add] 警告: 对话 '{conversation[:100]}...' 未生成任何粒子")
                    logger.warning(f"  可能原因: 1) 文本中未提取到实体 2) 实体提取失败 3) 情绪分析失败")
            except Exception as e:
                logger.error(f"[Amygdala.add] 粒子生成失败")
                logger.error(f"  对话内容: {conversation[:200]}{'...' if len(conversation) > 200 else ''}")
                logger.error(f"  conversation_id: {conversation_id}")
                logger.error(f"  错误信息: {str(e)}")
                import traceback
                logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
                particles = []
            
            if not particles:
                logger.warning(f"[Amygdala.add] 处理完成，但未生成粒子")
                logger.warning(f"  对话ID: {conversation_id}")
                logger.warning(f"  对话内容: {conversation[:200]}{'...' if len(conversation) > 200 else ''}")
                return {
                    "conversation_id": conversation_id,
                    "particles": [],
                    "particle_count": 0,
                    "relationship_map": {}
                }
            
            # Step 3: 构建邻域链接（可选）
            links_map: Optional[Dict[str, List[str]]] = None
            if self.auto_link_particles:
                try:
                    logger.info(f"[Amygdala.add] 开始构建粒子邻域链接")
                    logger.info(f"  参数: distance_threshold={self.link_distance_threshold}, top_k={self.link_top_k}")
                    links_map = auto_link_entities(
                        entities=particles,
                        projector=self.particle_projector,
                        distance_threshold=self.link_distance_threshold,
                        top_k=self.link_top_k
                    )
                    total_links = sum(len(links) for links in links_map.values())
                    logger.info(f"[Amygdala.add] 邻域链接构建完成: {total_links} 条链接")
                    if links_map:
                        for particle_id, links in links_map.items():
                            if links:
                                logger.debug(f"  粒子 {particle_id} 的邻居: {links}")
                except Exception as e:
                    logger.warning(f"[Amygdala.add] 邻域链接构建失败")
                    logger.warning(f"  错误信息: {str(e)}")
                    import traceback
                    logger.warning(f"  错误堆栈:\n{traceback.format_exc()}")
                    links_map = None
            
            # Step 4: 存储粒子到数据库
            try:
                logger.info(f"[Amygdala.add] 开始存储粒子到数据库")
                logger.info(f"  粒子数量: {len(particles)}")
                particle_ids = [p.entity_id for p in particles]
                logger.info(f"  粒子ID列表: {particle_ids}")
                
                self.particle_storage.upsert_entities(
                    entities=particles,
                    links_map=links_map
                )
                logger.info(f"[Amygdala.add] 粒子存储成功: {len(particles)} 个粒子已存储到数据库")
            except Exception as e:
                logger.error(f"[Amygdala.add] 粒子存储失败")
                logger.error(f"  对话ID: {conversation_id}")
                logger.error(f"  粒子数量: {len(particles)}")
                logger.error(f"  错误信息: {str(e)}")
                import traceback
                logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
                raise
            
            # Step 5: 存储对话到数据库
            try:
                logger.info(f"[Amygdala.add] 开始存储对话到数据库")
                logger.info(f"  conversation_id: {conversation_id}")
                logger.info(f"  对话内容长度: {len(conversation)} 字符")
                
                self.conversation_store.insert_strings([conversation])
                logger.info(f"[Amygdala.add] 对话存储成功: conversation_id={conversation_id}")
            except Exception as e:
                logger.error(f"[Amygdala.add] 对话存储失败")
                logger.error(f"  conversation_id: {conversation_id}")
                logger.error(f"  对话内容: {conversation[:200]}{'...' if len(conversation) > 200 else ''}")
                logger.error(f"  错误信息: {str(e)}")
                import traceback
                logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
                raise
            
            # Step 6: 记录粒子与对话的对应关系
            logger.info(f"[Amygdala.add] 开始记录关系映射")
            relationship_map: Dict[str, str] = {}
            for particle in particles:
                self.particle_to_conversation[particle.entity_id] = conversation_id
                relationship_map[particle.entity_id] = conversation_id
            
            # 更新反向映射
            if conversation_id not in self.conversation_to_particles:
                self.conversation_to_particles[conversation_id] = []
            self.conversation_to_particles[conversation_id].extend(
                [p.entity_id for p in particles]
            )
            
            logger.info(f"[Amygdala.add] 关系映射记录完成:")
            logger.info(f"  对话ID: {conversation_id}")
            logger.info(f"  关联粒子数: {len(relationship_map)}")
            logger.info(f"  关系映射详情: {relationship_map}")
            
            # 保存关系映射
            try:
                self._save_relationship_mapping()
                logger.debug(f"[Amygdala.add] 关系映射已保存到文件")
            except Exception as e:
                logger.warning(f"[Amygdala.add] 关系映射保存失败")
                logger.warning(f"  错误信息: {str(e)}")
            
            # 输出最终结果摘要
            logger.info(f"[Amygdala.add] 处理完成")
            logger.info(f"  对话ID: {conversation_id}")
            logger.info(f"  生成粒子数: {len(particles)}")
            logger.info(f"  粒子ID列表: {[p.entity_id for p in particles]}")
            logger.info(f"  关系映射数: {len(relationship_map)}")
            logger.info("=" * 80)
            
            return {
                "conversation_id": conversation_id,
                "particles": particles,
                "particle_count": len(particles),
                "relationship_map": relationship_map
            }
        
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"[Amygdala.add] 处理失败")
            logger.error(f"  对话ID: {conversation_id if 'conversation_id' in locals() else 'N/A'}")
            logger.error(f"  对话内容: {conversation[:200]}{'...' if len(conversation) > 200 else ''}")
            logger.error(f"  错误信息: {str(e)}")
            import traceback
            logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
            logger.error("=" * 80)
            raise
    
    def _save_relationship_mapping(self):
        """保存关系映射到文件"""
        if not self.particle_to_conversation:
            return
        
        particle_ids = list(self.particle_to_conversation.keys())
        conversation_ids = [self.particle_to_conversation[pid] for pid in particle_ids]
        
        df = pd.DataFrame({
            "particle_id": particle_ids,
            "conversation_id": conversation_ids
        })
        
        df.to_parquet(self.relationship_mapping_file, index=False)
        logger.debug(f"Saved relationship mapping to {self.relationship_mapping_file}")
    
    def _load_relationship_mapping(self):
        """从文件加载关系映射"""
        if not self.relationship_mapping_file.exists():
            logger.info("No existing relationship mapping file found")
            return
        
        try:
            df = pd.read_parquet(self.relationship_mapping_file)
            self.particle_to_conversation = {
                row["particle_id"]: row["conversation_id"]
                for _, row in df.iterrows()
            }
            
            # 构建反向映射
            self.conversation_to_particles = {}
            for particle_id, conversation_id in self.particle_to_conversation.items():
                if conversation_id not in self.conversation_to_particles:
                    self.conversation_to_particles[conversation_id] = []
                self.conversation_to_particles[conversation_id].append(particle_id)
            
            logger.info(
                f"Loaded {len(self.particle_to_conversation)} relationship mappings "
                f"for {len(self.conversation_to_particles)} conversations"
            )
        except Exception as e:
            logger.warning(f"Failed to load relationship mapping: {e}")
            self.particle_to_conversation = {}
            self.conversation_to_particles = {}
    
    def get_conversation_by_particle(self, particle_id: str) -> Optional[str]:
        """
        根据粒子 ID 获取对话 ID
        
        Args:
            particle_id: 粒子 ID
        
        Returns:
            对话 ID，如果不存在则返回 None
        """
        return self.particle_to_conversation.get(particle_id)
    
    def get_particles_by_conversation(self, conversation_id: str) -> List[str]:
        """
        根据对话 ID 获取粒子 ID 列表
        
        Args:
            conversation_id: 对话 ID
        
        Returns:
            粒子 ID 列表
        """
        return self.conversation_to_particles.get(conversation_id, []).copy()
    
    def get_conversation_text(self, conversation_id: str) -> Optional[str]:
        """
        根据对话 ID 获取对话文本
        
        Args:
            conversation_id: 对话 ID
        
        Returns:
            对话文本，如果不存在则返回 None
        """
        if hasattr(self.conversation_store, 'get_text'):
            return self.conversation_store.get_text(conversation_id)
        elif hasattr(self.conversation_store, 'hash_id_to_text'):
            return self.conversation_store.hash_id_to_text.get(conversation_id)
        else:
            # 尝试从 EmbeddingStore 获取
            try:
                row = self.conversation_store.get_row(conversation_id)
                return row.get('content') if row else None
            except:
                return None

    def retrieval(
        self,
        query_text: str,
        retrieval_mode: Literal["particle", "chunk"] = "particle",
        top_k: int = 10,
        cone_width: int = 50,
        max_neighbors: int = 20,
        neighbor_penalty: float = 1.1
    ) -> List[Dict[str, Any]]:
        """
        根据查询文本检索相关粒子或对话片段

        Args:
            query_text: 查询文本
            retrieval_mode: 检索模式
                - "particle": 返回 top-k 最相关的粒子，按相似度排序
                - "chunk": 返回粒子对应的对话片段（chunk），按包含的粒子数量和位置排序
            top_k: 返回结果数量
            cone_width: 锥体搜索宽度（建议值 50-100）
            max_neighbors: 邻域扩展时的最大节点数
            neighbor_penalty: 邻居惩罚系数（默认 1.1）

        Returns:
            retrieval_mode="particle" 时:
                [
                    {
                        "particle_id": str,
                        "entity": str,
                        "score": float,  # 双曲距离，越小越相似
                        "conversation_id": str,
                        "match_type": "direct" | "neighbor",
                        "metadata": dict
                    },
                    ...
                ]

            retrieval_mode="chunk" 时:
                [
                    {
                        "conversation_id": str,
                        "text": str,
                        "score": float,  # chunk 得分，包含越靠前的粒子越多，得分越高
                        "particle_count": int,  # 该 chunk 包含的粒子数量
                        "particle_ids": List[str],  # 包含的粒子 ID 列表
                    },
                    ...
                ]
        """
        from poincare.retrieval import HyperAmyRetrieval

        logger.info("=" * 80)
        logger.info(f"[Amygdala.retrieval] 开始检索")
        logger.info(f"  查询文本: {query_text[:200]}{'...' if len(query_text) > 200 else ''}")
        logger.info(f"  检索模式: {retrieval_mode}")
        logger.info(f"  top_k: {top_k}")

        # Step 1: 将查询文本转换为查询粒子
        logger.info(f"[Amygdala.retrieval] Step 1: 将查询文本转换为粒子...")
        query_particles = self.particle.process(
            text=query_text,
            text_id=f"query_{int(time.time())}"
        )

        if not query_particles:
            logger.warning(f"[Amygdala.retrieval] 查询文本未生成任何粒子，无法检索")
            return []

        # 使用第一个粒子作为查询粒子
        query_particle = query_particles[0]
        logger.info(f"[Amygdala.retrieval] 查询粒子: {query_particle.entity} (ID: {query_particle.entity_id})")

        # Step 2: 初始化检索器并执行检索
        logger.info(f"[Amygdala.retrieval] Step 2: 执行粒子检索...")
        retriever = HyperAmyRetrieval(
            storage=self.particle_storage,
            projector=self.particle_projector
        )

        search_results = retriever.search(
            query_entity=query_particle,
            top_k=top_k if retrieval_mode == "particle" else cone_width,
            cone_width=cone_width,
            max_neighbors=max_neighbors,
            neighbor_penalty=neighbor_penalty
        )

        if not search_results:
            logger.warning(f"[Amygdala.retrieval] 未检索到任何相关粒子")
            return []

        logger.info(f"[Amygdala.retrieval] 检索到 {len(search_results)} 个粒子")

        # Step 3: 根据检索模式返回结果
        if retrieval_mode == "particle":
            return self._format_particle_results(search_results, top_k)

        elif retrieval_mode == "chunk":
            return self._format_chunk_results(search_results, top_k)

        else:
            logger.error(f"[Amygdala.retrieval] 不支持的检索模式: {retrieval_mode}")
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}. Must be 'particle' or 'chunk'")

    def _format_particle_results(
        self,
        search_results: List['SearchResult'],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        格式化粒子检索结果

        Args:
            search_results: 检索结果列表
            top_k: 返回结果数量

        Returns:
            格式化的粒子结果列表
        """
        results = []

        for i, result in enumerate(search_results[:top_k]):
            # 获取粒子所属的对话 ID
            conversation_id = self.particle_to_conversation.get(result.id)

            particle_info = {
                "particle_id": result.id,
                "entity": result.metadata.get("entity", "Unknown"),
                "score": result.score,
                "conversation_id": conversation_id,
                "match_type": result.match_type,
                "metadata": {
                    "speed": result.metadata.get("v", 0.0),
                    "temperature": result.metadata.get("T", 0.0),
                    "born": result.metadata.get("born", 0.0),
                    "weight": result.metadata.get("weight", 1.0),
                }
            }
            results.append(particle_info)

            logger.debug(f"[Amygdala.retrieval] 粒子 {i+1}/{min(top_k, len(search_results))}: "
                        f"ID={result.id}, Entity={particle_info['entity']}, "
                        f"Score={result.score:.4f}, Match={result.match_type}")

        logger.info(f"[Amygdala.retrieval] 返回 {len(results)} 个粒子结果")
        return results

    def _format_chunk_results(
        self,
        search_results: List['SearchResult'],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        格式化 chunk 检索结果

        排序规则：
        - 一个 chunk 的得分取决于它包含的检索到的粒子
        - 包含的越靠前的粒子（在搜索结果中位置越靠前）越多，得分越高

        计算公式：
        chunk_score = sum(top_k - position) for each particle in chunk
        其中 position 是粒子在搜索结果中的位置（0-based）

        Args:
            search_results: 检索结果列表
            top_k: 返回结果数量

        Returns:
            格式化的 chunk 结果列表，按得分降序排序
        """
        # Step 1: 将粒子映射到对话（chunk）
        chunk_scores = {}  # {conversation_id: score}
        chunk_particle_counts = {}  # {conversation_id: count}
        chunk_particle_ids = {}  # {conversation_id: [particle_ids]}

        for position, result in enumerate(search_results):
            particle_id = result.id
            conversation_id = self.particle_to_conversation.get(particle_id)

            # 如果粒子不属于任何对话，跳过
            if not conversation_id:
                logger.debug(f"[Amygdala.retrieval] 粒子 {particle_id} 不属于任何对话，跳过")
                continue

            # 计算 chunk 得分：位置越靠前，贡献越大
            # 使用 (top_k - position) 作为权重，这样 top 个结果的权重最大
            weight = (len(search_results) - position)

            if conversation_id not in chunk_scores:
                chunk_scores[conversation_id] = 0
                chunk_particle_counts[conversation_id] = 0
                chunk_particle_ids[conversation_id] = []

            chunk_scores[conversation_id] += weight
            chunk_particle_counts[conversation_id] += 1
            chunk_particle_ids[conversation_id].append(particle_id)

        # Step 2: 按得分降序排序
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Step 3: 构建结果列表
        results = []

        for rank, (conversation_id, score) in enumerate(sorted_chunks):
            # 获取对话文本
            text = self.get_conversation_text(conversation_id)

            if not text:
                logger.warning(f"[Amygdala.retrieval] 对话 {conversation_id} 的文本未找到，跳过")
                continue

            chunk_info = {
                "conversation_id": conversation_id,
                "text": text,
                "score": score,
                "particle_count": chunk_particle_counts[conversation_id],
                "particle_ids": chunk_particle_ids[conversation_id],
                "rank": rank + 1
            }
            results.append(chunk_info)

            logger.info(f"[Amygdala.retrieval] Chunk {rank+1}/{len(sorted_chunks)}: "
                        f"ID={conversation_id}, Score={score:.1f}, "
                        f"Particles={chunk_particle_counts[conversation_id]}")

        logger.info(f"[Amygdala.retrieval] 返回 {len(results)} 个 chunk 结果")
        return results

