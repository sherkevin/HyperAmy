"""
轻量级 NER 模块

方案五：使用 spaCy 替代 LLM 进行实体抽取

性能：
- LLM 实体抽取：~2s
- spaCy NER：~50-100ms
- 加速比：20-40x
"""

import re
from typing import List
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LightweightNER:
    """
    轻量级命名实体识别（NER）类

    使用 spaCy 进行快速实体抽取，无需调用 LLM
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        初始化轻量级 NER

        Args:
            model_name: spaCy 模型名称
                - en_core_web_sm: 小模型，快速，准确率略低（推荐）
                - en_core_web_md: 中等模型，平衡速度和准确率
                - en_core_web_lg: 大模型，准确率最高，速度较慢
        """
        self.model_name = model_name

        try:
            import spacy
            self.nlp = spacy.load(model_name)
            logger.info(f"LightweightNER initialized with spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Downloading...")
            import spacy
            from spacy.cli import download

            try:
                download(model_name)
                self.nlp = spacy.load(model_name)
                logger.info(f"Downloaded and loaded spaCy model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                raise
        except ImportError:
            logger.error("spaCy not installed. Install with: pip install spacy")
            raise

        # 实体类型过滤（只保留重要的实体类型）
        self.relevant_entity_types = {
            'PERSON',      # 人名
            'ORG',         # 组织
            'GPE',         # 地理政治实体（国家、城市等）
            'LOC',         # 地理位置
            'PRODUCT',     # 产品
            'EVENT',       # 事件
            'WORK_OF_ART', # 艺术作品
            'LAW',         # 法律
            'LANGUAGE',    # 语言
            'NORP',        # 国籍、宗教、政治团体
            'FAC',         # 建筑、机场、高速公路等
        }

    def extract_entities(
        self,
        text: str,
        max_entities: int = 20,
        min_length: int = 2
    ) -> List[str]:
        """
        从文本中抽取实体

        Args:
            text: 输入文本
            max_entities: 最大实体数量（防止过多实体）
            min_length: 实体最小长度（过滤单字符）

        Returns:
            List[str]: 实体列表（去重后）
        """
        if not text or not text.strip():
            return []

        # 使用 spaCy 进行 NER
        doc = self.nlp(text)

        # 提取实体
        entities = set()

        for ent in doc.ents:
            # 过滤实体类型
            if ent.label_ in self.relevant_entity_types:
                entity_text = ent.text.strip()

                # 过滤短实体和常见停用词
                if len(entity_text) >= min_length and not self._is_stop_word(entity_text):
                    entities.add(entity_text)

        # 使用名词短语补充（spaCy 可能遗漏的重要名词）
        noun_phrases = self._extract_noun_phrases(doc, max_entities - len(entities))
        entities.update(noun_phrases)

        # 转换为列表并排序（按出现顺序）
        entity_list = sorted(list(entities), key=lambda x: text.find(x))

        # 限制数量
        entity_list = entity_list[:max_entities]

        logger.debug(f"[LightweightNER] Extracted {len(entity_list)} entities: {entity_list}")

        return entity_list

    def _extract_noun_phrases(self, doc, max_count: int = 10) -> List[str]:
        """
        提取名词短语作为补充实体

        Args:
            doc: spaCy Doc 对象
            max_count: 最大提取数量

        Returns:
            List[str]: 名词短语列表
        """
        noun_phrases = []

        for chunk in doc.noun_chunks:
            if len(noun_phrases) >= max_count:
                break

            phrase = chunk.text.strip()

            # 过滤条件
            if (
                len(phrase) >= 2 and
                not self._is_stop_word(phrase) and
                phrase not in noun_phrases
            ):
                noun_phrases.append(phrase)

        return noun_phrases

    def _is_stop_word(self, text: str) -> bool:
        """
        检查是否为停用词或无关词

        Args:
            text: 输入文本

        Returns:
            bool: True 表示是停用词
        """
        # 常见停用词列表
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'into', 'over', 'after', 'before', 'between',
            'under', 'again', 'there', 'here', 'up', 'down', 'out', 'off'
        }

        text_lower = text.lower().strip()
        return text_lower in stop_words or len(text_lower) < 2


class RuleBasedNER:
    """
    基于规则的 NER（备选方案）

    当 spaCy 不可用时，使用规则和正则表达式进行实体抽取
    """

    def __init__(self):
        logger.info("RuleBasedNER initialized (fallback mode)")

        # 常见实体模式
        self.patterns = [
            # 大写单词（人名、地名等）
            r'\b[A-Z][a-z]+\b',
            # 多词大写短语
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
            # 引号内容
            r'"([^"]+)"',
            # 专有名词（大写开头的多词短语）
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',
        ]

    def extract_entities(
        self,
        text: str,
        max_entities: int = 20,
        min_length: int = 2
    ) -> List[str]:
        """
        使用规则抽取实体

        Args:
            text: 输入文本
            max_entities: 最大实体数量
            min_length: 实体最小长度

        Returns:
            List[str]: 实体列表
        """
        if not text or not text.strip():
            return []

        entities = set()

        # 应用所有模式
        for pattern in self.patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""

                match = match.strip()

                if len(match) >= min_length and not self._is_stop_word(match):
                    entities.add(match)

        # 转换为列表并排序
        entity_list = sorted(list(entities), key=lambda x: text.find(x))
        entity_list = entity_list[:max_entities]

        logger.debug(f"[RuleBasedNER] Extracted {len(entity_list)} entities: {entity_list}")

        return entity_list

    def _is_stop_word(self, text: str) -> bool:
        """检查是否为停用词"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are'
        }
        return text.lower().strip() in stop_words or len(text) < 2
