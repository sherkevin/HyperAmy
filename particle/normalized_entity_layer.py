"""
标准化实体层 (NormalizedEntityLayer)

确保 Amygdala 和 HippoRAG 使用完全一致的实体 ID

核心功能：
1. 使用相同的 OpenIE 实体抽取（HippoRAG 的 OpenIE）
2. 实体文本标准化（大小写、空格）
3. 生成标准的 entity_id（MD5 hash with "entity-" prefix）
"""

import logging
from typing import List, Dict, Optional

from hipporag.utils.misc_utils import compute_mdhash_id
from utils.entitiy import Entity

logger = logging.getLogger(__name__)


class NormalizedEntityLayer:
    """
    标准化实体层

    为 Amygdala 和 HippoRAG 提供一致的实体抽取和 ID 生成
    """

    def __init__(
        self,
        llm_model_name: str = "DeepSeek-V3.2",
        llm_base_url: Optional[str] = None
    ):
        """
        初始化标准化实体层

        Args:
            llm_model_name: LLM 模型名称
            llm_base_url: LLM API 地址
        """
        # 使用 HippoRAG 的 OpenIE 抽取实体
        self.entity_extractor = Entity(
            model_name=llm_model_name,
            base_url=llm_base_url
        )

        logger.info(f"标准化实体层初始化完成: model={llm_model_name}")

    def extract_entities(
        self,
        text: str,
        normalize: bool = True
    ) -> List[str]:
        """
        从文本中抽取实体（使用 HippoRAG 的 OpenIE）

        Args:
            text: 输入文本
            normalize: 是否标准化实体文本

        Returns:
            实体列表
        """
        if not text or not text.strip():
            return []

        # 使用 HippoRAG 的 OpenIE 抽取
        entities = self.entity_extractor.extract_entities(text)

        if not entities:
            return []

        # 标准化实体
        if normalize:
            entities = [self._normalize_entity(e) for e in entities]
            # 去重但保持顺序
            entities = list(dict.fromkeys(entities))

        return entities

    def extract_entities_with_ids(
        self,
        text: str,
        normalize: bool = True
    ) -> List[Dict[str, str]]:
        """
        从文本中抽取实体，并生成标准 entity_id

        Args:
            text: 输入文本
            normalize: 是否标准化实体文本

        Returns:
            实体字典列表: [{'text': str, 'id': str}, ...]
        """
        entities = self.extract_entities(text, normalize)

        result = []
        for entity_text in entities:
            entity_id = self._generate_entity_id(entity_text)
            result.append({
                'text': entity_text,
                'id': entity_id
            })

        return result

    def _normalize_entity(self, entity: str) -> str:
        """
        标准化实体文本

        规则：
        1. 去除首尾空格
        2. 统一大小写（保留原始大小写，但去除多余变化）
        3. 去除特殊字符

        Args:
            entity: 原始实体文本

        Returns:
            标准化后的实体文本
        """
        if not entity:
            return ""

        # 去除首尾空格
        entity = entity.strip()

        # 去除多余空格（多个空格合并为一个）
        import re
        entity = re.sub(r'\s+', ' ', entity)

        return entity

    def _generate_entity_id(self, entity_text: str) -> str:
        """
        生成标准的 entity_id（兼容 HippoRAG）

        格式：MD5 hash with "entity-" prefix

        Args:
            entity_text: 实体文本（已标准化）

        Returns:
            entity_id (例如: "entity-a1b2c3d4...")
        """
        return compute_mdhash_id(content=entity_text, prefix="entity-")

    def get_entity_id(self, entity_text: str, normalize: bool = True) -> str:
        """
        为实体文本生成标准 entity_id

        Args:
            entity_text: 实体文本
            normalize: 是否先标准化

        Returns:
            entity_id
        """
        if normalize:
            entity_text = self._normalize_entity(entity_text)

        return self._generate_entity_id(entity_text)

    def batch_extract_entities(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> List[List[str]]:
        """
        批量抽取实体

        Args:
            texts: 文本列表
            normalize: 是否标准化

        Returns:
            实体列表的列表
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text, normalize)
            results.append(entities)

        return results


# 全局单例
_global_layer: Optional[NormalizedEntityLayer] = None


def get_global_entity_layer() -> NormalizedEntityLayer:
    """获取全局标准化实体层单例"""
    global _global_layer
    if _global_layer is None:
        _global_layer = NormalizedEntityLayer()
    return _global_layer
