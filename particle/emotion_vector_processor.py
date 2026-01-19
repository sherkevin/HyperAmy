"""
情绪向量处理器：将原始情绪向量转换为方向和模长

核心设计：
- 与情绪向量获取方式完全解耦（无论是LLM还是专门模型）
- 输入：原始情绪向量 e_raw
- 输出：方向 μ（归一化）+ 模长 κ（基于Soft Label强度）

公式：
- 方向：μ = e_raw / ||e_raw||
- 模长：κ = 1.0 + α × I_raw
- 其中 I_raw = max_{c ≠ neutral} (y_c)，α = 50.0

物理意义：
- 方向 μ：情绪的语义方向（如"愤怒-悲伤"轴）
- 模长 κ：情绪强度
  - κ ≈ 1-6：气态（弱情绪，快速遗忘）
  - κ ≈ 20-30：液态（中等情绪）
  - κ ≈ 40-50：固态（强情绪，慢速遗忘，类似"创伤"）
"""
import numpy as np
from typing import Union, List, Dict, Any
from dataclasses import dataclass
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedEmotionVector:
    """
    处理后的情绪向量

    Attributes:
        direction: 归一化方向向量 μ（单位向量）
        modulus: 模长 κ（基于情绪强度）
        intensity: 原始情绪强度 I_raw ∈ [0, 1]
        state: 物质状态描述（"气态"/"液态"/"固态"）
    """
    direction: np.ndarray  # 归一化方向 μ
    modulus: float         # 模长 κ
    intensity: float       # 情绪强度 I_raw
    state: str             # 状态描述

    def __repr__(self) -> str:
        return f"ProcessedEmotionVector(κ={self.modulus:.2f}, I={self.intensity:.3f}, state={self.state})"


class EmotionVectorProcessor:
    """
    情绪向量处理器

    功能：
    1. 将原始情绪向量归一化为方向 μ
    2. 根据 Soft Label 强度计算模长 κ
    3. 完全解耦情绪向量的来源
    """

    # 状态阈值
    STATE_GAS_MAX = 10.0      # κ < 10: 气态
    STATE_SOLID_MIN = 40.0    # κ >= 40: 固态
    # 中间是液态

    def __init__(
        self,
        alpha: float = 50.0,
        min_modulus: float = 1.0
    ):
        """
        初始化处理器

        Args:
            alpha: 模长系数 κ = 1.0 + α × I_raw（默认 50.0）
            min_modulus: 最小模长（确保 κ >= min_modulus）
        """
        self.alpha = alpha
        self.min_modulus = min_modulus
        logger.info(
            f"EmotionVectorProcessor initialized: "
            f"alpha={alpha}, min_modulus={min_modulus}"
        )

    def compute_state(self, modulus: float) -> str:
        """
        根据模长判断物质状态

        Args:
            modulus: 模长 κ

        Returns:
            str: 状态描述（"气态"/"液态"/"固态"）
        """
        if modulus < self.STATE_GAS_MAX:
            return "气态"
        elif modulus >= self.STATE_SOLID_MIN:
            return "固态"
        else:
            return "液态"

    def process(
        self,
        raw_vector: np.ndarray,
        intensity: float,
        normalize: bool = True
    ) -> ProcessedEmotionVector:
        """
        处理原始情绪向量

        Args:
            raw_vector: 原始情绪向量（来自任意来源）
            intensity: 情绪强度 I_raw ∈ [0, 1]
            normalize: 是否归一化方向（默认 True）

        Returns:
            ProcessedEmotionVector: 处理后的向量（方向 + 模长）
        """
        # 计算（归一化）方向
        if normalize:
            norm = np.linalg.norm(raw_vector)
            if norm > 1e-9:
                direction = raw_vector / norm
            else:
                logger.warning("Raw vector has near-zero norm, using as direction")
                direction = raw_vector.copy()
        else:
            direction = raw_vector.copy()

        # 计算模长：κ = 1.0 + α × I_raw
        modulus = self.min_modulus + self.alpha * intensity

        # 判断状态
        state = self.compute_state(modulus)

        logger.debug(
            f"Processed: I={intensity:.3f} → κ={modulus:.2f} ({state})"
        )

        return ProcessedEmotionVector(
            direction=direction,
            modulus=modulus,
            intensity=intensity,
            state=state
        )

    def process_batch(
        self,
        raw_vectors: List[np.ndarray],
        intensities: List[float],
        normalize: bool = True
    ) -> List[ProcessedEmotionVector]:
        """
        批量处理原始情绪向量

        Args:
            raw_vectors: 原始情绪向量列表
            intensities: 情绪强度列表
            normalize: 是否归一化方向

        Returns:
            List[ProcessedEmotionVector]: 处理后的向量列表
        """
        if len(raw_vectors) != len(intensities):
            raise ValueError(
                f"raw_vectors and intensities must have same length: "
                f"{len(raw_vectors)} != {len(intensities)}"
            )

        results = []
        for vec, intensity in zip(raw_vectors, intensities):
            processed = self.process(vec, intensity, normalize)
            results.append(processed)

        logger.info(
            f"Processed batch: {len(results)} vectors, "
            f"modulus range=[{min(r.modulus for r in results):.2f}, "
            f"{max(r.modulus for r in results):.2f}]"
        )

        return results

    def get_final_vector(self, processed: ProcessedEmotionVector) -> np.ndarray:
        """
        获取最终的向量（模长 × 方向）

        Args:
            processed: 处理后的向量

        Returns:
            np.ndarray: 最终向量 = κ × μ
        """
        return processed.direction * processed.modulus

    def get_final_vectors_batch(
        self,
        processed_list: List[ProcessedEmotionVector]
    ) -> List[np.ndarray]:
        """
        批量获取最终向量

        Args:
            processed_list: 处理后的向量列表

        Returns:
            List[np.ndarray]: 最终向量列表
        """
        return [self.get_final_vector(p) for p in processed_list]


# 全局单例
_global_processor: EmotionVectorProcessor = None


def get_global_processor(
    alpha: float = 50.0,
    min_modulus: float = 1.0
) -> EmotionVectorProcessor:
    """获取全局处理器单例"""
    global _global_processor
    if _global_processor is None:
        _global_processor = EmotionVectorProcessor(
            alpha=alpha,
            min_modulus=min_modulus
        )
    return _global_processor


def compute_direction_and_modulus(
    raw_vector: np.ndarray,
    intensity: float,
    alpha: float = 50.0
) -> tuple[np.ndarray, float]:
    """
    便捷函数：计算方向和模长

    Args:
        raw_vector: 原始情绪向量
        intensity: 情绪强度 I_raw ∈ [0, 1]
        alpha: 模长系数

    Returns:
        tuple: (方向 μ, 模长 κ)
    """
    processor = get_global_processor(alpha=alpha)
    processed = processor.process(raw_vector, intensity)
    return processed.direction, processed.modulus
