"""
Emotion Cache Manager

优化方案三：缓存常见实体的情感描述和嵌入向量

功能：
1. 缓存情感描述生成结果
2. 缓存嵌入向量
3. 支持持久化存储
4. 自动管理缓存大小
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmotionCache:
    """
    情感缓存管理器

    缓存策略：
    - 使用 (sentence, entity) 作为缓存键
    - 支持情感描述和嵌入向量的缓存
    - 基于文件系统的持久化存储
    """

    def __init__(self, cache_dir: str = "./emotion_cache"):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # 创建子目录
        self.desc_cache_dir = self.cache_dir / "descriptions"
        self.embed_cache_dir = self.cache_dir / "embeddings"

        self.desc_cache_dir.mkdir(exist_ok=True)
        self.embed_cache_dir.mkdir(exist_ok=True)

        # 缓存统计
        self.stats = {
            "description_hits": 0,
            "description_misses": 0,
            "embedding_hits": 0,
            "embedding_misses": 0
        }

        logger.info(f"EmotionCache initialized with cache_dir: {cache_dir}")

    def _get_cache_key(self, sentence: str, entity: str) -> str:
        """
        生成缓存键

        Args:
            sentence: 原始句子
            entity: 实体名称

        Returns:
            str: MD5哈希值作为缓存键
        """
        content = f"{sentence}|{entity}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get_cached_description(self, sentence: str, entity: str) -> Optional[str]:
        """
        获取缓存的情感描述

        Args:
            sentence: 原始句子
            entity: 实体名称

        Returns:
            Optional[str]: 缓存的情感描述，如果不存在则返回None
        """
        cache_key = self._get_cache_key(sentence, entity)
        cache_file = self.desc_cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # 检查缓存是否过期（可选，这里设置为永久有效）
                # 如果需要添加过期时间，可以在cached_data中添加timestamp字段

                self.stats["description_hits"] += 1
                logger.debug(f"Cache HIT for description: entity='{entity}', cache_key={cache_key}")
                return cached_data["description"]
            except Exception as e:
                logger.warning(f"Failed to load cached description for '{entity}': {e}")
                # 删除损坏的缓存文件
                cache_file.unlink(missing_ok=True)

        self.stats["description_misses"] += 1
        logger.debug(f"Cache MISS for description: entity='{entity}', cache_key={cache_key}")
        return None

    def save_description(self, sentence: str, entity: str, description: str):
        """
        保存情感描述到缓存

        Args:
            sentence: 原始句子
            entity: 实体名称
            description: 情感描述
        """
        cache_key = self._get_cache_key(sentence, entity)
        cache_file = self.desc_cache_dir / f"{cache_key}.pkl"

        try:
            cached_data = {
                "sentence": sentence,
                "entity": entity,
                "description": description,
                "timestamp": time.time()
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)

            logger.debug(f"Saved description to cache: entity='{entity}', cache_key={cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save description to cache for '{entity}': {e}")

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        获取缓存的嵌入向量

        Args:
            text: 输入文本

        Returns:
            Optional[np.ndarray]: 缓存的嵌入向量，如果不存在则返回None
        """
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = self.embed_cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                self.stats["embedding_hits"] += 1
                logger.debug(f"Cache HIT for embedding: text='{text[:50]}...', cache_key={cache_key}")
                return cached_data["embedding"]
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                # 删除损坏的缓存文件
                cache_file.unlink(missing_ok=True)

        self.stats["embedding_misses"] += 1
        logger.debug(f"Cache MISS for embedding: text='{text[:50]}...', cache_key={cache_key}")
        return None

    def save_embedding(self, text: str, embedding: np.ndarray):
        """
        保存嵌入向量到缓存

        Args:
            text: 输入文本
            embedding: 嵌入向量
        """
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = self.embed_cache_dir / f"{cache_key}.pkl"

        try:
            cached_data = {
                "text": text,
                "embedding": embedding,
                "timestamp": time.time()
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)

            logger.debug(f"Saved embedding to cache: text='{text[:50]}...', cache_key={cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            Dict: 包含命中率和未命中率的统计信息
        """
        total_desc = self.stats["description_hits"] + self.stats["description_misses"]
        total_embed = self.stats["embedding_hits"] + self.stats["embedding_misses"]

        desc_hit_rate = (self.stats["description_hits"] / total_desc * 100) if total_desc > 0 else 0
        embed_hit_rate = (self.stats["embedding_hits"] / total_embed * 100) if total_embed > 0 else 0

        return {
            "description": {
                "hits": self.stats["description_hits"],
                "misses": self.stats["description_misses"],
                "hit_rate": desc_hit_rate
            },
            "embedding": {
                "hits": self.stats["embedding_hits"],
                "misses": self.stats["embedding_misses"],
                "hit_rate": embed_hit_rate
            }
        }

    def clear_cache(self):
        """
        清空所有缓存
        """
        import shutil

        for cache_file in self.desc_cache_dir.glob("*.pkl"):
            cache_file.unlink()

        for cache_file in self.embed_cache_dir.glob("*.pkl"):
            cache_file.unlink()

        logger.info("Cleared all cache files")

        # 重置统计
        self.stats = {
            "description_hits": 0,
            "description_misses": 0,
            "embedding_hits": 0,
            "embedding_misses": 0
        }

    def get_cache_size(self) -> Dict[str, int]:
        """
        获取缓存大小

        Returns:
            Dict: 包含各种缓存文件数量的字典
        """
        desc_count = len(list(self.desc_cache_dir.glob("*.pkl")))
        embed_count = len(list(self.embed_cache_dir.glob("*.pkl")))

        return {
            "description_cache_files": desc_count,
            "embedding_cache_files": embed_count,
            "total_files": desc_count + embed_count
        }
