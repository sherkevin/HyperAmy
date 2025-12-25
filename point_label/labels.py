"""
Labels 类

计算记忆深度和温度：输入 chunk，输出 emotion vector、记忆深度和温度
- 记忆深度 = f(emotion embed 纯度, 模长)
- 温度 = f(纯度, 困惑度)，表示情绪波动程度
"""
import numpy as np
import json
import re
from dataclasses import dataclass
from typing import Optional, List
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import create_client
from llm.config import API_URL_CHAT, API_URL_COMPLETIONS, DEFAULT_MODEL
from hipporag.utils.logging_utils import get_logger
from prompts.prompt_template_manager import PromptTemplateManager

logger = get_logger(__name__)

# 定义情绪列表（30种情绪）
EMOTIONS = [
    # 基本情绪（Plutchik的8种基本情绪）
    "joy",           # 快乐
    "sadness",       # 悲伤
    "anger",         # 愤怒
    "fear",          # 恐惧
    "surprise",      # 惊讶
    "disgust",       # 厌恶
    "trust",         # 信任
    "anticipation",  # 期待
    
    # 扩展情绪
    "love",          # 爱
    "hate",          # 恨
    "anxiety",       # 焦虑
    "calm",          # 平静
    "excitement",    # 兴奋
    "disappointment", # 失望
    "pride",         # 骄傲
    "shame",         # 羞耻
    "guilt",         # 愧疚
    "relief",         # 解脱
    "hope",          # 希望
    "despair",       # 绝望
    "contentment",   # 满足
    "frustration",   # 沮丧
    "gratitude",     # 感激
    "resentment",    # 怨恨
    "loneliness",    # 孤独
    "nostalgia",     # 怀旧
    "envy",          # 嫉妒
    "contempt",      # 轻蔑
]


@dataclass
class LabelsResult:
    """Labels 提取结果"""
    emotion_vector: np.ndarray  # 情感向量（30维）
    memory_depth: float  # 记忆深度（0~1）
    temperature: Optional[float] = None  # 温度（0~1），如果使用 specific 模式则计算，否则为 None


class Labels:
    """
    记忆深度计算类
    
    功能：输入 chunk，输出记忆深度（0~1标量）
    记忆深度 = 纯度 × 模长归一化
    - 纯度：单个情绪分量占比，越大表示情绪越纯
    - 模长：情绪强度，保留原始模长信息
    """
    
    def __init__(self, model_name=None, magnitude_scale=10.0, perplexity_scale=50.0):
        """
        初始化 Labels 类
        
        Args:
            model_name: 使用的模型名称，如果为 None 则使用默认模型
            magnitude_scale: 模长归一化的缩放因子，用于将模长映射到合理范围
            perplexity_scale: 困惑度归一化的缩放因子，用于将困惑度映射到合理范围
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.magnitude_scale = magnitude_scale
        self.perplexity_scale = perplexity_scale
        
        # 创建 LLM 客户端（使用 normal 模式，Chat API）
        self.client = create_client(
            model_name=self.model_name,
            chat_api_url=API_URL_CHAT,
            mode="normal"
        )
        
        # 创建 specific 模式的客户端（用于获取 token 概率）
        self.specific_client = create_client(
            model_name=self.model_name,
            api_url=API_URL_COMPLETIONS,
            mode="specific"
        )
        
        # 初始化 prompt 模板管理器
        self.prompt_template_manager = PromptTemplateManager()
        
        logger.info(
            f"Labels initialized with model: {self.model_name}, "
            f"magnitude_scale: {magnitude_scale}, perplexity_scale: {perplexity_scale}"
        )
    
    def _extract_emotion_vector(self, chunk: str) -> np.ndarray:
        """
        提取 chunk 的情感向量（不归一化）
        
        Args:
            chunk: 输入文本片段
        
        Returns:
            numpy.ndarray: 原始情感向量 (30维)，不归一化
        """
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
            
            # 构建向量（按照EMOTIONS的顺序），不归一化
            vector = np.array([emotion_dict.get(emotion, 0.0) for emotion in EMOTIONS])
            
            return vector
            
        except Exception as e:
            logger.error(f"Failed to extract emotion vector: {e}")
            raise
    
    def _compute_purity(self, vector: np.ndarray) -> float:
        """
        计算 emotion vector 的纯度
        
        纯度 = max(emotion_vector) / sum(emotion_vector)
        值越大表示单个情绪分量占比越大，情绪越纯
        
        Args:
            vector: 情感向量
        
        Returns:
            float: 纯度值 (0~1)
        """
        vector_sum = np.sum(vector)
        if vector_sum == 0:
            return 0.0
        
        max_component = np.max(vector)
        purity = max_component / vector_sum
        
        return float(purity)
    
    def _normalize_magnitude(self, norm: float) -> float:
        """
        将模长归一化到 0~1 范围
        
        使用 tanh 函数将模长映射到合理范围
        模长越大，归一化后的值越大，但不会超过 1
        
        Args:
            norm: L2 模长
        
        Returns:
            float: 归一化后的模长 (0~1)
        """
        # 使用 tanh 将模长映射到 0~1
        # magnitude_scale 控制映射的敏感度
        normalized = np.tanh(norm / self.magnitude_scale)
        return float(normalized)
    
    def _compute_perplexity(self, chunk: str) -> float:
        """
        计算 chunk 的困惑度（perplexity）
        
        困惑度 = exp(mean(-logprob))
        困惑度越低，表示模型越确定（温度低）
        困惑度越高，表示模型越不确定（温度高）
        
        Args:
            chunk: 输入文本片段
        
        Returns:
            float: 困惑度值
        """
        try:
            # 使用 specific 模式获取 token 概率
            result = self.specific_client.complete(
                query=chunk,
                echo=True,  # 回显 prompt，获取 chunk 的 token 概率
                max_tokens=1,  # 设置一个小的值，但我们只使用 prompt_tokens
                temperature=0.0,  # 温度设为 0 保证确定性
                logprobs=1  # 获取 top-1 的 logprob
            )
            
            # 获取 prompt 部分的 token 信息（即 chunk 的 token）
            prompt_tokens = result.prompt_tokens
            
            if not prompt_tokens:
                logger.warning(f"No tokens found for chunk: {chunk[:50]}...")
                return float('inf')  # 返回无穷大表示无法计算
            
            # 提取 logprobs（如果为 None，则设为负无穷）
            logprobs = [
                token.logprob if token.logprob is not None else float('-inf')
                for token in prompt_tokens
            ]
            
            # 过滤掉无效的 logprob
            valid_logprobs = [lp for lp in logprobs if lp != float('-inf')]
            
            if not valid_logprobs:
                logger.warning(f"All token logprobs are None for chunk: {chunk[:50]}...")
                return float('inf')
            
            # 计算平均 logprob
            mean_logprob = np.mean(valid_logprobs)
            
            # 计算困惑度：perplexity = exp(-mean_logprob)
            perplexity = np.exp(-mean_logprob)
            
            return float(perplexity)
            
        except Exception as e:
            logger.error(f"Failed to compute perplexity: {e}")
            return float('inf')
    
    def _normalize_perplexity(self, perplexity: float) -> float:
        """
        将困惑度归一化到 0~1 范围
        
        使用 tanh 函数将困惑度映射到合理范围
        困惑度越大，归一化后的值越大，但不会超过 1
        
        Args:
            perplexity: 困惑度值
        
        Returns:
            float: 归一化后的困惑度 (0~1)
        """
        # 使用 tanh 将困惑度映射到 0~1
        # perplexity_scale 控制映射的敏感度
        normalized = np.tanh(perplexity / self.perplexity_scale)
        return float(normalized)
    
    def _compute_temperature(self, purity: float, perplexity: float) -> float:
        """
        计算温度（情绪波动程度）
        
        温度 = f(纯度, 困惑度)
        - 如果纯度纯（高）且困惑度低 → 温度低（情绪纯且稳定）
        - 如果纯度低或困惑度高 → 温度高（情绪波动大）
        
        温度 = (1 - 纯度) + 归一化困惑度，然后归一化到 0~1
        
        Args:
            purity: 纯度值 (0~1)
            perplexity: 困惑度值
        
        Returns:
            float: 温度值 (0~1)，越大表示情绪波动越大
        """
        # 归一化困惑度
        normalized_perplexity = self._normalize_perplexity(perplexity)
        
        # 温度 = (1 - 纯度) + 归一化困惑度
        # 纯度越低或困惑度越高，温度越高
        temperature = (1.0 - purity) + normalized_perplexity
        
        # 归一化到 0~1（理论上最大值为 2，但实际不会达到）
        temperature = min(temperature / 2.0, 1.0)
        
        return float(temperature)
    
    def extract(self, chunk: str, use_specific: bool = False) -> LabelsResult:
        """
        提取 chunk 的情感向量、记忆深度和温度
        
        记忆深度 = 纯度 × 模长归一化
        - 纯度：单个情绪分量占比，越大表示情绪越纯（值得记忆）
        - 模长：情绪强度，保留原始模长信息
        
        温度 = f(纯度, 困惑度)
        - 如果纯度纯（高）且困惑度低 → 温度低（情绪纯且稳定）
        - 如果纯度低或困惑度高 → 温度高（情绪波动大）
        - 仅在 use_specific=True 时计算
        
        Args:
            chunk: 输入文本片段
            use_specific: 是否使用 specific 模式计算温度
        
        Returns:
            LabelsResult: 包含 emotion_vector, memory_depth, temperature 的结果对象
        """
        # 提取情感向量（不归一化）
        emotion_vector = self._extract_emotion_vector(chunk)
        
        # 计算纯度（单个情绪分量占比）
        purity = self._compute_purity(emotion_vector)
        
        # 计算模长
        magnitude = np.linalg.norm(emotion_vector)
        
        # 归一化模长到 0~1
        normalized_magnitude = self._normalize_magnitude(magnitude)
        
        # 记忆深度 = 纯度 × 模长归一化
        memory_depth = purity * normalized_magnitude
        
        # 计算温度（仅在 use_specific=True 时）
        temperature = None
        if use_specific:
            perplexity = self._compute_perplexity(chunk)
            if perplexity != float('inf'):
                temperature = self._compute_temperature(purity, perplexity)
                logger.debug(
                    f"Chunk temperature: {temperature:.4f} "
                    f"(purity={purity:.4f}, perplexity={perplexity:.4f})"
                )
            else:
                logger.warning(f"Failed to compute perplexity, temperature set to None")
        
        logger.debug(
            f"Chunk memory depth: {memory_depth:.4f} "
            f"(purity={purity:.4f}, magnitude={magnitude:.4f}, normalized_magnitude={normalized_magnitude:.4f})"
        )
        
        return LabelsResult(
            emotion_vector=emotion_vector,
            memory_depth=float(memory_depth),
            temperature=temperature
        )

