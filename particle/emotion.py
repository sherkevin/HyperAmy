"""
Emotion 类

简洁的情感向量提取：输入 chunk，输出 emotion vector
支持缓存、批量提取和质量验证。
"""
import numpy as np
import json
import re
from typing import Optional, List

from llm import create_client
from llm.config import API_URL_CHAT, DEFAULT_MODEL
from hipporag.utils.logging_utils import get_logger
from prompts.prompt_template_manager import PromptTemplateManager

logger = get_logger(__name__)

# 定义情绪列表（30种情绪）
# EMOTIONS = [
#     # 基本情绪（Plutchik的8种基本情绪）
#     "joy",           # 快乐
#     "sadness",       # 悲伤
#     "anger",         # 愤怒
#     "fear",          # 恐惧
#     "surprise",      # 惊讶
#     "disgust",       # 厌恶
#     "trust",         # 信任
#     "anticipation",  # 期待
    
#     # 扩展情绪
#     "love",          # 爱
#     "hate",          # 恨
#     "anxiety",       # 焦虑
#     "calm",          # 平静
#     "excitement",    # 兴奋
#     "disappointment", # 失望
#     "pride",         # 骄傲
#     "shame",         # 羞耻
#     "guilt",         # 愧疚
#     "relief",         # 解脱
#     "hope",          # 希望
#     "despair",       # 绝望
#     "contentment",   # 满足
#     "frustration",   # 沮丧
#     "gratitude",     # 感激
#     "resentment",    # 怨恨
#     "loneliness",    # 孤独
#     "nostalgia",     # 怀旧
#     "envy",          # 嫉妒
#     "contempt",      # 轻蔑
# ]

EMOTIONS = [
    # Positive
    "admiration", "amusement", "approval", "caring", "desire", 
    "excitement", "gratitude", "joy", "love", "optimism", 
    "pride", "relief",
    
    # Negative
    "anger", "annoyance", "disappointment", "disapproval", "disgust", 
    "embarrassment", "fear", "grief", "nervousness", "remorse", 
    "sadness", 
    
    # Ambiguous / Cognitive
    "confusion", "curiosity", "realization", "surprise",
    
    # Neutral
    "neutral"
]


class Emotion:
    """
    情感向量提取类
    
    功能：输入 chunk，输出 emotion vector
    支持缓存、批量提取和质量验证。
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        初始化 Emotion 类
        
        Args:
            model_name: 使用的模型名称，如果为 None 则使用默认模型
            enable_cache: 是否启用缓存（默认True）
            cache_dir: 缓存目录（如果不提供，使用默认路径）
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.enable_cache = enable_cache
        
        # 创建 LLM 客户端（使用 normal 模式，Chat API）
        self.client = create_client(
            model_name=self.model_name,
            chat_api_url=API_URL_CHAT,
            mode="normal"
        )
        
        # 初始化 prompt 模板管理器
        self.prompt_template_manager = PromptTemplateManager()
        
        # 初始化缓存（如果启用）
        self.cache = None
        if self.enable_cache:
            try:
                from utils.cache import EmotionVectorCache
                self.cache = EmotionVectorCache(cache_dir=cache_dir, use_persistent=True)
                logger.info(f"Emotion cache enabled: {cache_dir or '.cache/emotion_vectors'}")
            except ImportError:
                logger.warning("Cache module not available, caching disabled")
                self.enable_cache = False
        
        logger.info(f"Emotion initialized with model: {self.model_name}, cache={'enabled' if self.enable_cache else 'disabled'}")
    
    def extract(self, chunk: str, validate: bool = True) -> np.ndarray:
        """
        提取 chunk 的情感向量
        
        Args:
            chunk: 输入文本片段
            validate: 是否验证提取结果的质量（默认True）
        
        Returns:
            numpy.ndarray: 归一化后的情感向量 (30维)
            
        Raises:
            ValueError: 如果chunk为空或无效
            RuntimeError: 如果提取失败且无法fallback
        """
        # 输入验证
        if not isinstance(chunk, str):
            raise ValueError(f"chunk must be a string, got {type(chunk)}")
        if len(chunk.strip()) < 5:
            logger.warning(f"Chunk too short ({len(chunk)} chars), may produce poor results")
        
        # 检查缓存
        if self.enable_cache and self.cache is not None:
            cached_vector = self.cache.get(chunk)
            if cached_vector is not None:
                logger.debug(f"Cache hit for emotion extraction")
                return cached_vector
        
        emotions_str = ", ".join(EMOTIONS)
        
        # 使用模板管理器渲染 prompt
        prompt = self.prompt_template_manager.render(
            name='emotion_extraction',
            emotions_list=emotions_str,
            chunk=chunk
        )

        try:
            # 使用 CompletionClient 调用 API
            result = self.client.complete(
                query=prompt,
                max_tokens=500,
                temperature=0.2  # 低温度保证一致性
            )
            
            content = result.get_answer_text().strip()
            
            # 提取JSON（可能包含markdown代码块）
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
            
            try:
                emotion_dict = json.loads(json_str)
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试提取数字
                logger.warning(f"Failed to parse JSON, attempting to extract values from: {content[:200]}")
                emotion_dict = {}
                for emotion in EMOTIONS:
                    pattern = f'"{emotion}"\\s*:\\s*([0-9.]+)'
                    match = re.search(pattern, content)
                    if match:
                        emotion_dict[emotion] = float(match.group(1))
                    else:
                        emotion_dict[emotion] = 0.0
            
            # 构建向量（按照EMOTIONS的顺序）
            vector = np.array([emotion_dict.get(emotion, 0.0) for emotion in EMOTIONS])
            
            # L2归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            else:
                # 如果向量全为0，返回均匀分布
                vector = np.ones(len(EMOTIONS)) / len(EMOTIONS)
            
            # 质量验证
            if validate:
                quality_score = self._validate_emotion_vector(vector)
                if quality_score < 0.3:
                    logger.warning(f"Low quality emotion vector extracted (quality={quality_score:.3f}), chunk length={len(chunk)}")
            
            # 缓存结果
            if self.enable_cache and self.cache is not None:
                self.cache.set(chunk, vector)
            
            return vector
            
        except Exception as e:
            logger.error(f"Failed to extract emotion vector for chunk (length={len(chunk)}): {e}")
            # 返回零向量作为fallback
            fallback_vector = np.zeros(len(EMOTIONS))
            return fallback_vector
    
    def _validate_emotion_vector(self, vector: np.ndarray) -> float:
        """
        验证情绪向量的质量
        
        Args:
            vector: 情绪向量
            
        Returns:
            质量分数（0-1之间，越高越好）
        """
        if vector is None or len(vector) != len(EMOTIONS):
            return 0.0
        
        # 计算质量指标
        # 1. 非零元素比例（应该有明显的情绪分布）
        non_zero_ratio = np.sum(np.abs(vector) > 1e-6) / len(vector)
        
        # 2. 最大值（应该有主导情绪）
        max_value = np.max(vector)
        
        # 3. 熵（应该有合理的多样性，不能太集中也不能太分散）
        vector_positive = np.clip(vector, 1e-10, None)  # 避免log(0)
        entropy = -np.sum(vector_positive * np.log(vector_positive))
        max_entropy = np.log(len(EMOTIONS))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # 综合质量分数
        quality_score = (non_zero_ratio * 0.3 + max_value * 0.4 + normalized_entropy * 0.3)
        
        return float(quality_score)
    
    def extract_batch(
        self,
        chunks: List[str],
        validate: bool = True,
        max_workers: int = 10
    ) -> List[np.ndarray]:
        """
        批量提取情绪向量（并发处理）
        
        Args:
            chunks: 文本片段列表
            validate: 是否验证提取结果
            max_workers: 并发线程数
            
        Returns:
            情绪向量列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        results = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.extract, chunk, validate): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(chunks), desc="Extracting emotions"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Failed to extract emotion for chunk {idx}: {e}")
                    results[idx] = np.zeros(len(EMOTIONS))  # Fallback
        
        return results

