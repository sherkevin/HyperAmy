"""
统一实体抽取服务 (UnifiedEntityExtractor)

目标：为 Amygdala 和 HippoRAG 提供一致的实体抽取，解决 GraphFusion 融合失效问题

核心设计：
1. 使用 LLM 抽取高质量实体（保留 Amygdala 的优势）
2. 使用标准的 entity_id 格式（兼容 HippoRAG: MD5 hash with "entity-" prefix）
3. 提供统一的 API 接口
4. 支持缓存优化

使用示例：
    >>> from workflow.unified_entity_extractor import UnifiedEntityExtractor
    >>>
    >>> extractor = UnifiedEntityExtractor()
    >>>
    >>> # 抽取实体
    >>> entities = extractor.extract_entities(
    ...     text="Count Monte Cristo refused the grapes.",
    ...     text_id="doc_001"
    ... )
    >>>
    >>> print(entities)
    >>> [
    >>>     {'text': 'Count Monte Cristo', 'id': 'entity-a1b2c3d4...'},
    >>>     {'text': 'grapes', 'id': 'entity-e5f6g7h8...'}
    >>> ]
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from llm.config import API_KEY, BASE_URL
from hipporag.utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class UnifiedEntityExtractor:
    """
    统一实体抽取器

    为 Amygdala 和 HippoRAG 提供一致的实体抽取服务
    """

    def __init__(
        self,
        llm_model_name: str = "DeepSeek-V3.2",
        llm_base_url: str = BASE_URL,
        enable_cache: bool = True,
        cache_dir: str = "./cache/unified_entities"
    ):
        """
        初始化统一实体抽取器

        Args:
            llm_model_name: LLM 模型名称
            llm_base_url: LLM API 地址
            enable_cache: 是否启用缓存
            cache_dir: 缓存目录
        """
        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir)

        # 创建缓存目录
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 延迟导入（避免循环依赖）
        self._llm_client = None
        self._ner = None

        logger.info(f"统一实体抽取器初始化完成: model={llm_model_name}")

    @property
    def llm_client(self):
        """延迟初始化 LLM 客户端"""
        if self._llm_client is None:
            from llm.llm import LLM
            self._llm_client = LLM(
                model_name=self.llm_model_name,
                base_url=self.llm_base_url
            )
        return self._llm_client

    @property
    def ner(self):
        """延迟初始化 NER"""
        if self._ner is None:
            from utils.entitiy import Entity
            self._ner = Entity()
        return self._ner

    def extract_entities(
        self,
        text: str,
        text_id: Optional[str] = None,
        use_llm: bool = True,
        max_entities: int = 50
    ) -> List[str]:
        """
        从文本中抽取实体

        Args:
            text: 输入文本
            text_id: 文本 ID（用于缓存）
            use_llm: 是否使用 LLM 抽取（False 则使用规则方法）
            max_entities: 最大实体数量

        Returns:
            实体列表（按出现顺序）
        """
        if not text or not text.strip():
            return []

        # 检查缓存
        if self.enable_cache and text_id:
            cached = self._load_from_cache(text_id)
            if cached is not None:
                logger.debug(f"从缓存加载实体: text_id={text_id}, count={len(cached)}")
                return cached

        # 抽取实体
        if use_llm:
            entities = self._extract_with_llm(text, max_entities)
        else:
            entities = self._extract_with_rules(text)

        # 标准化实体（大写、去除多余空格）
        entities = [self._normalize_entity(e) for e in entities if e]
        entities = list(dict.fromkeys(entities))  # 去重但保持顺序

        # 保存到缓存
        if self.enable_cache and text_id:
            self._save_to_cache(text_id, entities)

        logger.debug(f"抽取到 {len(entities)} 个实体: {entities[:10]}")

        return entities

    def extract_entities_with_ids(
        self,
        text: str,
        text_id: Optional[str] = None,
        use_llm: bool = True,
        max_entities: int = 50
    ) -> List[Dict[str, str]]:
        """
        从文本中抽取实体，并生成标准 entity_id

        Args:
            text: 输入文本
            text_id: 文本 ID
            use_llm: 是否使用 LLM
            max_entities: 最大实体数量

        Returns:
            实体字典列表: [{'text': str, 'id': str}, ...]
        """
        entities = self.extract_entities(text, text_id, use_llm, max_entities)

        result = []
        for entity_text in entities:
            entity_id = compute_mdhash_id(content=entity_text, prefix="entity-")
            result.append({
                'text': entity_text,
                'id': entity_id
            })

        return result

    def _extract_with_llm(self, text: str, max_entities: int) -> List[str]:
        """
        使用 LLM 抽取实体

        策略：使用 HippoRAG 的 NER 模块（基于 LLM）
        """
        try:
            # 使用 HippoRAG 的 Entity 抽取
            entities = self.ner.extract(text)

            if not entities:
                logger.warning(f"LLM 实体抽取返回空，使用规则方法")
                return self._extract_with_rules(text)

            return entities[:max_entities]

        except Exception as e:
            logger.error(f"LLM 实体抽取失败: {e}，使用规则方法")
            return self._extract_with_rules(text)

    def _extract_with_rules(self, text: str) -> List[str]:
        """
        使用规则方法抽取实体（备用方案）

        策略：
        1. 提取专有名词（大写开头的词）
        2. 提取引号中的词组
        3. 提取常见实体模式
        """
        entities = set()

        # 规则 1：提取大写开头的词（专有名词）
        # 例如：Count, Mercedes, Monte Cristo
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(proper_nouns)

        # 规则 2：提取引号中的词组
        # 例如："sacred tie", "East"
        quoted_phrases = re.findall(r'"([^"]+)"', text)
        for phrase in quoted_phrases:
            # 分割并保留多词短语
            words = phrase.split()
            if len(words) > 1:
                entities.add(phrase)
            entities.update(words)

        # 规则 3：常见实体模式（小写但有特定上下文）
        # 例如：Count of Monte Cristo 中的 Count
        pattern_matches = re.findall(
            r'\b(Count|Duke|King|Queen|Prince|Princess|Sir|Madame|Mr|Mrs|Ms|Dr)\b',
            text,
            flags=re.IGNORECASE
        )
        entities.update([m.title() for m in pattern_matches])

        # 过滤常见停用词
        stopwords = {
            'The', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To',
            'For', 'Of', 'With', 'By', 'From', 'As', 'Is', 'Was', 'Are'
        }
        entities = entities - stopwords

        # 按文本出现顺序排序
        entities_list = list(entities)
        entities_list.sort(key=lambda e: text.find(e) if text.find(e) != -1 else len(text))

        return entities_list

    def _normalize_entity(self, entity: str) -> str:
        """
        标准化实体文本

        - 去除多余空格
        - 统一大小写（专有名词保留首字母大写）
        """
        if not entity:
            return ""

        # 去除多余空格
        entity = ' '.join(entity.split())

        # 保留原始大小写（LLM 已经处理好了）
        return entity

    def _get_cache_path(self, text_id: str) -> Path:
        """获取缓存文件路径"""
        # 使用 MD5 hash 作为文件名
        import hashlib
        hash_id = hashlib.md5(text_id.encode()).hexdigest()
        return self.cache_dir / f"{hash_id}.txt"

    def _load_from_cache(self, text_id: str) -> Optional[List[str]]:
        """从缓存加载实体"""
        if not self.enable_cache:
            return None

        cache_path = self._get_cache_path(text_id)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                entities = [line.strip() for line in f if line.strip()]
            return entities
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return None

    def _save_to_cache(self, text_id: str, entities: List[str]):
        """保存实体到缓存"""
        if not self.enable_cache:
            return

        cache_path = self._get_cache_path(text_id)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                for entity in entities:
                    f.write(f"{entity}\n")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def get_entity_id(self, entity_text: str) -> str:
        """
        为实体文本生成标准 entity_id

        兼容 HippoRAG 格式: MD5 hash with "entity-" prefix

        Args:
            entity_text: 实体文本

        Returns:
            entity_id (例如: "entity-a1b2c3d4...")
        """
        return compute_mdhash_id(content=entity_text, prefix="entity-")

    def batch_extract_entities(
        self,
        texts: List[str],
        text_ids: Optional[List[str]] = None,
        use_llm: bool = True,
        max_entities: int = 50
    ) -> List[List[str]]:
        """
        批量抽取实体

        Args:
            texts: 文本列表
            text_ids: 文本 ID 列表（可选）
            use_llm: 是否使用 LLM
            max_entities: 最大实体数量

        Returns:
            实体列表的列表
        """
        if text_ids is None:
            text_ids = [f"batch_{i}" for i in range(len(texts))]

        results = []
        for text, text_id in zip(texts, text_ids):
            entities = self.extract_entities(text, text_id, use_llm, max_entities)
            results.append(entities)

        return results


# 全局单例
_global_extractor: Optional[UnifiedEntityExtractor] = None


def get_global_extractor() -> UnifiedEntityExtractor:
    """获取全局统一实体抽取器单例"""
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = UnifiedEntityExtractor()
    return _global_extractor


def extract_entities(
    text: str,
    text_id: Optional[str] = None,
    use_llm: bool = True
) -> List[str]:
    """
    便捷函数：抽取实体

    Args:
        text: 输入文本
        text_id: 文本 ID
        use_llm: 是否使用 LLM

    Returns:
        实体列表
    """
    extractor = get_global_extractor()
    return extractor.extract_entities(text, text_id, use_llm)


def get_entity_id(entity_text: str) -> str:
    """
    便捷函数：生成 entity_id

    Args:
        entity_text: 实体文本

    Returns:
        entity_id
    """
    return compute_mdhash_id(content=entity_text, prefix="entity-")
