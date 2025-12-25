"""
存储层模块

负责 Point 对象的持久化存储，使用 ChromaDB 作为底层存储。
"""
import logging
import torch
import chromadb
from .types import Point

logger = logging.getLogger(__name__)


class HyperAmyStorage:
    """
    存储层：负责 Point 对象的持久化
    
    使用 ChromaDB 作为底层存储，支持向量相似度搜索和元数据过滤。
    """
    def __init__(self, persist_path="./hyperamy_db"):
        """
        Args:
            persist_path: ChromaDB 数据库持久化路径
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name="emotion_particles",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度进行向量搜索
        )

    def upsert_point(self, point: Point):
        """
        存储或更新粒子
        
        Args:
            point: 要存储的 Point 对象
            
        Raises:
            Exception: 存储失败时抛出异常
        """
        try:
            # 1. 向量归一化：确保存储的是归一化向量，纯粹表示"方向"
            norm_vec = torch.nn.functional.normalize(point.emotion_vector, p=2, dim=-1)
            embedding_list = norm_vec.tolist()

            # 2. 准备 Metadata (调用 Point 的封装方法)
            metadata = point.to_metadata()

            # 3. 写入 ChromaDB
            self.collection.upsert(
                ids=[point.id],
                embeddings=[embedding_list],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Failed to upsert point {point.id}: {str(e)}")
            raise e

