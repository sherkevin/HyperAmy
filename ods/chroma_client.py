"""
ChromaDB 数据访问层

封装 ChromaDB 的所有底层操作，提供统一的数据访问接口。
"""
import json
import logging
import numpy as np
import torch
import chromadb
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChromaClient:
    """
    ChromaDB 客户端封装类
    
    提供统一的数据库操作接口，处理向量归一化、metadata 序列化等细节。
    """
    
    def __init__(self, persist_path: str = "./hyperamy_db", collection_name: str = "emotion_particles"):
        """
        初始化 ChromaDB 客户端
        
        Args:
            persist_path: 数据库持久化路径
            collection_name: 集合名称
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度进行向量搜索
        )
        logger.info(f"ChromaClient initialized: path={persist_path}, collection={collection_name}")
    
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        插入或更新向量和元数据
        
        Args:
            ids: 文档 ID 列表
            embeddings: 向量列表（每个向量是一个浮点数列表）
            metadatas: 元数据列表（可选）
            
        Raises:
            Exception: 操作失败时抛出异常
        """
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.debug(f"Upserted {len(ids)} items")
        except Exception as e:
            logger.error(f"Failed to upsert items: {str(e)}")
            raise e
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        查询相似向量
        
        Args:
            query_embeddings: 查询向量列表
            n_results: 返回结果数量
            where: 元数据过滤条件（可选）
            include: 要包含的字段列表，如 ["metadatas", "embeddings", "documents"]
        
        Returns:
            查询结果字典，包含 ids, embeddings, metadatas, distances 等字段
        """
        try:
            result = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include or ["metadatas", "embeddings"]
            )
            logger.debug(f"Query returned {len(result.get('ids', [[]])[0]) if result.get('ids') else 0} results")
            return result
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise e
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        根据 ID 获取文档
        
        Args:
            ids: 文档 ID 列表（可选）
            where: 元数据过滤条件（可选）
            include: 要包含的字段列表
        
        Returns:
            文档数据字典，包含 ids, embeddings, metadatas 等字段
        """
        try:
            result = self.collection.get(
                ids=ids,
                where=where,
                include=include or ["metadatas", "embeddings"]
            )
            logger.debug(f"Get returned {len(result.get('ids', []))} items")
            return result
        except Exception as e:
            logger.error(f"Get failed: {str(e)}")
            raise e
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        删除文档
        
        Args:
            ids: 要删除的文档 ID 列表（可选）
            where: 元数据过滤条件（可选）
        """
        try:
            self.collection.delete(ids=ids, where=where)
            logger.debug(f"Deleted items: ids={ids}, where={where}")
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            raise e
    
    @staticmethod
    def normalize_vector(vec: np.ndarray) -> List[float]:
        """
        归一化向量并转换为列表
        
        Args:
            vec: numpy 数组或 torch.Tensor
        
        Returns:
            归一化后的向量列表
        """
        # 转换为 torch.Tensor 进行归一化
        if isinstance(vec, np.ndarray):
            vec_tensor = torch.from_numpy(vec).float()
        elif isinstance(vec, torch.Tensor):
            vec_tensor = vec.float()
        else:
            raise TypeError(f"Unsupported vector type: {type(vec)}")
        
        # L2 归一化
        norm_vec = torch.nn.functional.normalize(vec_tensor, p=2, dim=-1)
        return norm_vec.tolist()
    
    @staticmethod
    def serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        序列化元数据，将列表和字典转换为 JSON 字符串
        
        ChromaDB 的 metadata 只支持 int, float, str, bool 类型。
        
        Args:
            metadata: 原始元数据字典
        
        Returns:
            序列化后的元数据字典
        """
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                serialized[key] = json.dumps(value)
            else:
                serialized[key] = value
        return serialized
    
    @staticmethod
    def deserialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        反序列化元数据，将 JSON 字符串转换回列表或字典
        
        Args:
            metadata: 序列化后的元数据字典
        
        Returns:
            反序列化后的元数据字典
        """
        deserialized = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                try:
                    # 尝试解析 JSON 字符串
                    parsed = json.loads(value)
                    if isinstance(parsed, (list, dict)):
                        deserialized[key] = parsed
                    else:
                        deserialized[key] = value
                except (json.JSONDecodeError, TypeError):
                    # 不是 JSON 字符串，保持原样
                    deserialized[key] = value
            else:
                deserialized[key] = value
        return deserialized

