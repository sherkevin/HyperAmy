"""
Emotion Classifier: 将情绪描述转换为 Soft Label

输出格式：{emotion: probability}
包含情绪类别和对应的概率值，用于计算情绪强度 I_raw
"""
from typing import Dict, List
import re
import numpy as np
from llm import create_client
from llm.config import API_URL_CHAT, DEFAULT_MODEL
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 情绪类别定义（不含 neutral）
EMOTION_CATEGORIES = [
    "joy", "happiness", "happy", "excited", "enthusiasm", "cheerful",
    "sadness", "sad", "sorrow", "grief", "melancholy", "depressed",
    "anger", "angry", "rage", "furious", "irritated", "annoyed",
    "fear", "afraid", "terrified", "anxious", "worried", "nervous",
    "surprise", "surprised", "amazed", "shocked", "astonished",
    "disgust", "disgusted", "revolted", "repulsed",
    "love", "affection", "tenderness", "warmth", "fondness",
    "hate", "hatred", "contempt", "loathing", "dislike",
    "interest", "curiosity", "intrigue", "engaged",
    "boredom", "bored", "indifferent", "uninterested",
    "pride", "proud", "confident", "satisfied",
    "shame", "ashamed", "embarrassed", "guilty",
    "calm", "peaceful", "serene", "relaxed", "tranquil",
    "tense", "stressed", "uneasy", "restless",
    "hope", "hopeful", "optimistic", "expectant",
    "despair", "hopeless", "pessimistic", "desperate"
]

# 中性情绪词
NEUTRAL_WORDS = [
    "neutral", "calm", "peaceful", "serene", "relaxed", "tranquil",
    "indifferent", "uninterested", "bored"
]

# 情绪到主要类别的映射
EMOTION_MAPPING = {
    # joy
    "joy": "joy", "happiness": "joy", "happy": "joy", "excited": "joy",
    "enthusiasm": "joy", "cheerful": "joy", "delight": "joy", "elated": "joy",
    # sadness
    "sadness": "sadness", "sad": "sadness", "sorrow": "sadness",
    "grief": "sadness", "melancholy": "sadness", "depressed": "sadness",
    # anger
    "anger": "anger", "angry": "anger", "rage": "anger", "furious": "anger",
    "irritated": "anger", "annoyed": "anger", "frustrated": "anger",
    # fear
    "fear": "fear", "afraid": "fear", "terrified": "fear",
    "anxious": "fear", "worried": "fear", "nervous": "fear", "scared": "fear",
    # surprise
    "surprise": "surprise", "surprised": "surprise", "amazed": "surprise",
    "shocked": "surprise", "astonished": "surprise",
    # disgust
    "disgust": "disgust", "disgusted": "disgust", "revolted": "disgust",
    "repulsed": "disgust",
    # love
    "love": "love", "affection": "love", "tenderness": "love",
    "warmth": "love", "fondness": "love", "adoring": "love",
    # hate
    "hate": "hate", "hatred": "hate", "contempt": "hate",
    "loathing": "hate", "dislike": "hate",
    # interest
    "interest": "interest", "curiosity": "interest", "intrigue": "interest",
    "engaged": "interest", "fascinated": "interest",
    # boredom
    "boredom": "boredom", "bored": "boredom", "indifferent": "boredom",
    "uninterested": "boredom",
    # pride
    "pride": "pride", "proud": "pride", "confident": "pride",
    "satisfied": "pride",
    # shame
    "shame": "shame", "ashamed": "shame", "embarrassed": "shame",
    "guilty": "shame",
    # calm
    "calm": "neutral", "peaceful": "neutral", "serene": "neutral",
    "relaxed": "neutral", "tranquil": "neutral",
    # tense
    "tense": "fear", "stressed": "fear", "uneasy": "fear",
    "restless": "fear",
    # hope
    "hope": "joy", "hopeful": "joy", "optimistic": "joy",
    "expectant": "interest",
    # despair
    "despair": "sadness", "hopeless": "sadness", "pessimistic": "sadness",
    "desperate": "sadness"
}

# 主要情绪类别
MAIN_CATEGORIES = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust", "love", "hate", "interest"]


class EmotionClassifier:
    """
    情绪分类器：将情绪描述转换为 Soft Label

    输入：情绪描述文本，如 "enthusiasm, contentment, camaraderie"
    输出：Soft Label {emotion: probability}

    情绪强度 I_raw = max_{c != neutral} (y_c)
    """

    def __init__(self, model_name=None):
        """
        初始化情绪分类器

        Args:
            model_name: LLM 模型名称（用于更复杂的分类）
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.client = None  # 延迟初始化
        logger.info(f"EmotionClassifier initialized with model: {self.model_name}")

    def _get_client(self):
        """延迟初始化 LLM 客户端"""
        if self.client is None:
            self.client = create_client(
                model_name=self.model_name,
                chat_api_url=API_URL_CHAT,
                mode="normal"
            )
        return self.client

    def classify_rule_based(self, text: str) -> Dict[str, float]:
        """
        基于规则的情绪分类（快速方法）

        从文本中提取情绪词，根据预定义映射生成 soft label

        Args:
            text: 情绪描述文本

        Returns:
            Dict[str, float]: Soft label {category: probability}
        """
        # 初始化所有类别为 0
        soft_label = {cat: 0.0 for cat in MAIN_CATEGORIES}

        if not text or not text.strip():
            return soft_label

        # 转换为小写并分割
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # 统计情绪词
        emotion_counts = {cat: 0 for cat in MAIN_CATEGORIES}
        total_count = 0

        for word in words:
            # 检查完整词匹配
            if word in EMOTION_MAPPING:
                mapped_cat = EMOTION_MAPPING[word]
                if mapped_cat in emotion_counts:
                    emotion_counts[mapped_cat] += 1
                    total_count += 1

        # 如果没有找到情绪词，返回全 neutral
        if total_count == 0:
            soft_label["neutral"] = 1.0
            return soft_label

        # 归一化为概率分布
        for cat in MAIN_CATEGORIES:
            soft_label[cat] = emotion_counts[cat] / total_count

        return soft_label

    def classify_llm_based(self, text: str) -> Dict[str, float]:
        """
        基于 LLM 的情绪分类（更准确但较慢）

        Args:
            text: 情绪描述文本

        Returns:
            Dict[str, float]: Soft label {category: probability}
        """
        prompt = f"""Analyze the emotion(s) in the following text and output a probability distribution over these emotion categories:

Categories: {', '.join(MAIN_CATEGORIES)}

Text: "{text}"

Output format: JSON with keys as category names and values as probabilities (sum = 1.0).
Example: {{"joy": 0.6, "neutral": 0.3, "interest": 0.1, ...}}

Output only the JSON, no other text:"""

        try:
            client = self._get_client()
            result = client.complete(
                query=prompt,
                max_tokens=200,
                temperature=0.3
            )

            import json
            response_text = result.get_answer_text().strip()

            # 尝试解析 JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            soft_label = json.loads(response_text.strip())

            # 确保所有类别都存在
            for cat in MAIN_CATEGORIES:
                if cat not in soft_label:
                    soft_label[cat] = 0.0

            # 归一化
            total = sum(soft_label.values())
            if total > 0:
                soft_label = {k: v / total for k, v in soft_label.items()}

            return soft_label

        except Exception as e:
            logger.warning(f"LLM-based classification failed: {e}, falling back to rule-based")
            return self.classify_rule_based(text)

    def classify(
        self,
        text: str,
        use_llm: bool = False
    ) -> Dict[str, float]:
        """
        分类情绪描述为 soft label

        Args:
            text: 情绪描述文本
            use_llm: 是否使用 LLM 分类（默认 False，使用规则）

        Returns:
            Dict[str, float]: Soft label {category: probability}
        """
        if use_llm:
            return self.classify_llm_based(text)
        else:
            return self.classify_rule_based(text)

    def get_intensity(self, text: str, use_llm: bool = False) -> float:
        """
        获取情绪强度 I_raw

        I_raw = max_{c != neutral} (y_c)

        Args:
            text: 情绪描述文本
            use_llm: 是否使用 LLM 分类

        Returns:
            float: 情绪强度 [0, 1]
        """
        soft_label = self.classify(text, use_llm=use_llm)

        # 获取除 neutral 外的最大概率
        max_non_neutral = 0.0
        for cat, prob in soft_label.items():
            if cat != "neutral":
                max_non_neutral = max(max_non_neutral, prob)

        return max_non_neutral

    def get_intensity_batch(
        self,
        texts: List[str],
        use_llm: bool = False
    ) -> List[float]:
        """
        批量获取情绪强度

        Args:
            texts: 情绪描述文本列表
            use_llm: 是否使用 LLM 分类

        Returns:
            List[float]: 情绪强度列表
        """
        intensities = []
        for text in texts:
            intensity = self.get_intensity(text, use_llm=use_llm)
            intensities.append(intensity)

        logger.info(f"Computed intensities for {len(intensities)} texts: {intensities}")
        return intensities


# 全局单例
_global_classifier: EmotionClassifier = None


def get_global_classifier() -> EmotionClassifier:
    """获取全局情绪分类器单例"""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = EmotionClassifier()
    return _global_classifier


def compute_intensity(text: str, use_llm: bool = False) -> float:
    """
    便捷函数：计算情绪强度

    Args:
        text: 情绪描述文本
        use_llm: 是否使用 LLM 分类

    Returns:
        float: 情绪强度 [0, 1]
    """
    classifier = get_global_classifier()
    return classifier.get_intensity(text, use_llm=use_llm)
