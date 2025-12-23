"""
情感向量存储模块

类似 EmbeddingStore，用于存储和管理情感向量。
"""
import numpy as np
import os
from typing import List, Dict, Optional
import pandas as pd
import sys

# 添加 hipporag 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'src'))
from hipporag.utils.misc_utils import compute_mdhash_id
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmotionStore:
    """情感向量存储类，类似 EmbeddingStore"""
    
    def __init__(self, emotion_extractor, db_filename, batch_size, namespace):
        """
        初始化情感向量存储
        
        Args:
            emotion_extractor: EmotionExtractor 实例
            db_filename: 存储目录
            batch_size: 批处理大小
            namespace: 命名空间（用于区分不同的存储）
        """
        self.emotion_extractor = emotion_extractor
        self.batch_size = batch_size
        self.namespace = namespace
        
        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)
        
        self.filename = os.path.join(
            db_filename, f"emotion_{self.namespace}.parquet"
        )
        self._load_data()
    
    def _load_data(self):
        """加载已存储的情感向量"""
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["content"].values.tolist()
            # 将列表形式的向量转换回 numpy array
            self.emotion_vectors = [np.array(v) for v in df["emotion_vector"].values.tolist()]
            
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t, "emotion_vector": v}
                for h, t, v in zip(self.hash_ids, self.texts, self.emotion_vectors)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            
            assert len(self.hash_ids) == len(self.texts) == len(self.emotion_vectors)
            logger.info(f"Loaded {len(self.hash_ids)} emotion vectors from {self.filename}")
        else:
            self.hash_ids, self.texts, self.emotion_vectors = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}
            self.hash_id_to_text, self.text_to_hash_id = {}, {}
    
    def _save_data(self):
        """保存情感向量到文件"""
        # 将 numpy array 转换为列表以便存储
        emotion_vectors_list = [v.tolist() for v in self.emotion_vectors]
        
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "emotion_vector": emotion_vectors_list
        })
        data_to_save.to_parquet(self.filename, index=False)
        
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t, "emotion_vector": v}
            for h, t, v in zip(self.hash_ids, self.texts, self.emotion_vectors)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        
        logger.info(f"Saved {len(self.hash_ids)} emotion vectors to {self.filename}")
    
    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict:
        """获取缺失的文本的 hash_id"""
        nodes_dict = {}
        
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}
        
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return {}
        
        existing = self.hash_id_to_row.keys()
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        
        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}
    
    def insert_strings(self, texts: List[str]):
        """
        插入文本并提取情感向量
        
        Args:
            texts: 文本列表
        """
        nodes_dict = {}
        
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}
        
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return
        
        existing = self.hash_id_to_row.keys()
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        
        logger.info(
            f"Inserting {len(missing_ids)} new emotion vectors, "
            f"{len(all_hash_ids) - len(missing_ids)} already exist."
        )
        
        if not missing_ids:
            return  # All records already exist
        
        # 准备需要提取情感的文本
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        
        # 批量提取情感向量
        try:
            missing_emotion_vectors, _ = self.emotion_extractor.batch_extract_emotion_vectors(
                texts_to_encode, batch_size=self.batch_size
            )
        except Exception as e:
            logger.error(f"Failed to extract emotion vectors: {e}")
            # 如果提取失败，使用零向量
            missing_emotion_vectors = [np.zeros(len(self.emotion_extractor.extract_emotion_vector("test")[0])) 
                                      for _ in texts_to_encode]
        
        self._upsert(missing_ids, texts_to_encode, missing_emotion_vectors)
    
    def _upsert(self, hash_ids, texts, emotion_vectors):
        """更新或插入情感向量"""
        self.emotion_vectors.extend(emotion_vectors)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        
        logger.info(f"Saving new emotion vectors.")
        self._save_data()
    
    def get_emotion_vector(self, hash_id, dtype=np.float32) -> np.ndarray:
        """获取指定 hash_id 的情感向量"""
        return self.emotion_vectors[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_emotion_vectors(self, hash_ids, dtype=np.float32) -> List[np.ndarray]:
        """批量获取情感向量"""
        if not hash_ids:
            return []
        
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        emotion_vectors = np.array(self.emotion_vectors, dtype=dtype)[indices]
        
        return emotion_vectors.tolist()
    
    def get_row(self, hash_id):
        """获取指定 hash_id 的行数据"""
        return self.hash_id_to_row[hash_id]
    
    def get_hash_id(self, text):
        """获取文本的 hash_id"""
        return self.text_to_hash_id.get(text)
    
    def get_all_ids(self):
        """获取所有 hash_id"""
        return self.hash_ids.copy()
    
    def get_all_id_to_rows(self):
        """获取所有 hash_id 到行的映射"""
        return self.hash_id_to_row.copy()

