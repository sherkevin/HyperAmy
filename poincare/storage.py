"""
存储层模块

负责 ParticleEntity 对象的持久化存储，作为调用层调用 ods 层。
"""
import logging
import time
import json
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from ods import ChromaClient

if TYPE_CHECKING:
    # 仅用于类型检查，避免运行时导入
    from particle import ParticleEntity

logger = logging.getLogger(__name__)


class HyperAmyStorage:
    """
    存储层：负责 ParticleEntity 对象的持久化（调用层）
    
    作为业务层和 ods 层之间的桥梁，处理业务逻辑到数据访问的转换。
    """
    def __init__(self, persist_path: Optional[str] = None, collection_name: str = "emotion_particles"):
        """
        Args:
            persist_path: 数据库持久化路径。如果为 None，则根据 collection_name 自动生成。
            collection_name: 集合名称
        """
        # 如果未提供 persist_path，则根据 collection_name 隐式生成
        if persist_path is None:
            persist_path = f"./hyperamy_db_{collection_name}"
        
        self.ods_client = ChromaClient(persist_path=persist_path, collection_name=collection_name)
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
                "links": json.dumps(links or [])  # 序列化为 JSON 字符串
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
                metadata = {
                    "v": float(entity.speed),
                    "T": float(entity.temperature),
                    "weight": float(entity.weight),  # 粒子质量
                    "born": float(entity.born),
                    "last_updated": float(time.time()),
                    "text_id": entity.text_id,
                    "entity": entity.entity,
                    "links": json.dumps(links)
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

