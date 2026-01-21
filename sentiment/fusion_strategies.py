#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分数融合策略模块

实现多种语义分数和情绪分数的融合策略，包括线性加权、调和平均、几何平均和基于排名的融合。
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class NormalizationStrategy(Enum):
    """归一化策略枚举"""
    MIN_MAX = "min_max"  # Min-Max归一化
    Z_SCORE = "z_score"  # Z-score归一化
    NONE = "none"  # 不归一化
    L2 = "l2"  # L2归一化


class FusionStrategy(Enum):
    """融合策略枚举"""
    LINEAR = "linear"  # 线性加权（默认）
    HARMONIC = "harmonic"  # 调和平均
    GEOMETRIC = "geometric"  # 几何平均
    RANK_FUSION = "rank_fusion"  # 基于排名的融合（RRF）


class ScoreNormalizer:
    """分数归一化器"""
    
    @staticmethod
    def normalize(
        scores: np.ndarray,
        strategy: NormalizationStrategy = NormalizationStrategy.MIN_MAX
    ) -> np.ndarray:
        """
        归一化分数数组
        
        Args:
            scores: 原始分数数组
            strategy: 归一化策略
            
        Returns:
            归一化后的分数数组
        """
        if len(scores) == 0:
            return scores
        
        scores = np.array(scores, dtype=float)
        
        if strategy == NormalizationStrategy.NONE:
            return scores
        elif strategy == NormalizationStrategy.MIN_MAX:
            return ScoreNormalizer._min_max_normalize(scores)
        elif strategy == NormalizationStrategy.Z_SCORE:
            return ScoreNormalizer._z_score_normalize(scores)
        elif strategy == NormalizationStrategy.L2:
            return ScoreNormalizer._l2_normalize(scores)
        else:
            logger.warning(f"Unknown normalization strategy: {strategy}, using MIN_MAX")
            return ScoreNormalizer._min_max_normalize(scores)
    
    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        """Min-Max归一化到[0, 1]"""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-8:
            return np.ones_like(scores) * 0.5  # 如果所有值相同，返回0.5
        return (scores - min_val) / (max_val - min_val)
    
    @staticmethod
    def _z_score_normalize(scores: np.ndarray) -> np.ndarray:
        """Z-score归一化"""
        mean = scores.mean()
        std = scores.std()
        if std < 1e-8:
            return np.zeros_like(scores)  # 如果标准差为0，返回0
        normalized = (scores - mean) / std
        # 映射到[0, 1]范围（假设3-sigma规则）
        normalized = (normalized + 3) / 6
        return np.clip(normalized, 0, 1)
    
    @staticmethod
    def _l2_normalize(scores: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norm = np.linalg.norm(scores)
        if norm < 1e-8:
            return scores
        return scores / norm


class ScoreFusion:
    """分数融合器"""
    
    @staticmethod
    def fuse(
        semantic_scores: np.ndarray,
        emotion_scores: np.ndarray,
        sentiment_weight: float = 0.5,
        strategy: FusionStrategy = FusionStrategy.LINEAR,
        semantic_weight: Optional[float] = None,
        normalization_strategy: NormalizationStrategy = NormalizationStrategy.MIN_MAX
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        融合语义分数和情绪分数
        
        Args:
            semantic_scores: 语义相似度分数数组
            emotion_scores: 情绪相似度分数数组
            sentiment_weight: 情绪权重（0-1之间）
            strategy: 融合策略
            semantic_weight: 语义权重（如果不提供，自动计算为1-sentiment_weight）
            normalization_strategy: 归一化策略
            
        Returns:
            (融合后的分数数组, 融合信息字典)
        """
        assert len(semantic_scores) == len(emotion_scores), \
            f"分数数组长度不匹配: {len(semantic_scores)} vs {len(emotion_scores)}"
        
        # 确保权重在有效范围内
        sentiment_weight = np.clip(sentiment_weight, 0.0, 1.0)
        if semantic_weight is None:
            semantic_weight = 1.0 - sentiment_weight
        else:
            semantic_weight = np.clip(semantic_weight, 0.0, 1.0)
        
        # 归一化分数
        semantic_normalized = ScoreNormalizer.normalize(
            semantic_scores, normalization_strategy
        )
        emotion_normalized = ScoreNormalizer.normalize(
            emotion_scores, normalization_strategy
        )
        
        # 根据策略融合
        if strategy == FusionStrategy.LINEAR:
            fused_scores = ScoreFusion._linear_fusion(
                semantic_normalized, emotion_normalized, semantic_weight, sentiment_weight
            )
        elif strategy == FusionStrategy.HARMONIC:
            fused_scores = ScoreFusion._harmonic_fusion(
                semantic_normalized, emotion_normalized, semantic_weight, sentiment_weight
            )
        elif strategy == FusionStrategy.GEOMETRIC:
            fused_scores = ScoreFusion._geometric_fusion(
                semantic_normalized, emotion_normalized, semantic_weight, sentiment_weight
            )
        elif strategy == FusionStrategy.RANK_FUSION:
            fused_scores = ScoreFusion._rank_fusion(
                semantic_scores, emotion_scores  # 使用原始分数进行排名
            )
        else:
            logger.warning(f"Unknown fusion strategy: {strategy}, using LINEAR")
            fused_scores = ScoreFusion._linear_fusion(
                semantic_normalized, emotion_normalized, semantic_weight, sentiment_weight
            )
        
        # 构建融合信息
        fusion_info = {
            'strategy': strategy.value,
            'normalization': normalization_strategy.value,
            'semantic_weight': semantic_weight,
            'sentiment_weight': sentiment_weight,
            'semantic_mean': float(np.mean(semantic_normalized)),
            'emotion_mean': float(np.mean(emotion_normalized)),
            'fused_mean': float(np.mean(fused_scores))
        }
        
        return fused_scores, fusion_info
    
    @staticmethod
    def _linear_fusion(
        semantic_scores: np.ndarray,
        emotion_scores: np.ndarray,
        semantic_weight: float,
        sentiment_weight: float
    ) -> np.ndarray:
        """线性加权融合"""
        return semantic_weight * semantic_scores + sentiment_weight * emotion_scores
    
    @staticmethod
    def _harmonic_fusion(
        semantic_scores: np.ndarray,
        emotion_scores: np.ndarray,
        semantic_weight: float,
        sentiment_weight: float
    ) -> np.ndarray:
        """调和平均融合"""
        # 避免除零
        semantic_part = semantic_weight / (semantic_scores + 1e-8)
        emotion_part = sentiment_weight / (emotion_scores + 1e-8)
        
        harmonic_scores = (semantic_weight + sentiment_weight) / (semantic_part + emotion_part)
        
        # 处理无效值
        harmonic_scores = np.nan_to_num(harmonic_scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        return harmonic_scores
    
    @staticmethod
    def _geometric_fusion(
        semantic_scores: np.ndarray,
        emotion_scores: np.ndarray,
        semantic_weight: float,
        sentiment_weight: float
    ) -> np.ndarray:
        """几何平均融合"""
        # 避免零值和负值
        semantic_safe = np.maximum(semantic_scores, 1e-8)
        emotion_safe = np.maximum(emotion_scores, 1e-8)
        
        # 加权几何平均
        geometric_scores = np.power(
            semantic_safe, semantic_weight
        ) * np.power(
            emotion_safe, sentiment_weight
        )
        
        return geometric_scores
    
    @staticmethod
    def _rank_fusion(
        semantic_scores: np.ndarray,
        emotion_scores: np.ndarray,
        k: int = 60
    ) -> np.ndarray:
        """
        基于排名的融合（Reciprocal Rank Fusion, RRF）
        
        RRF公式: RRF(d) = Σ 1/(k + rank_i(d))
        其中rank_i(d)是文档d在第i个排序中的排名，k是常数（通常取60）
        
        Args:
            semantic_scores: 语义分数数组
            emotion_scores: 情绪分数数组
            k: RRF常数（默认60）
            
        Returns:
            融合后的RRF分数数组
        """
        # 获取排名（分数越高排名越靠前，排名从1开始）
        semantic_ranks = ScoreFusion._get_ranks(semantic_scores)
        emotion_ranks = ScoreFusion._get_ranks(emotion_scores)
        
        # 计算RRF分数
        rrf_scores = (
            1.0 / (k + semantic_ranks) +
            1.0 / (k + emotion_ranks)
        )
        
        return rrf_scores
    
    @staticmethod
    def _get_ranks(scores: np.ndarray) -> np.ndarray:
        """
        获取分数对应的排名（分数越高排名越靠前，排名从1开始）
        
        Args:
            scores: 分数数组
            
        Returns:
            排名数组
        """
        # 使用argsort获取排序索引（降序）
        sorted_indices = np.argsort(scores)[::-1]
        
        # 创建排名数组
        ranks = np.zeros_like(scores, dtype=int)
        for rank, idx in enumerate(sorted_indices, start=1):
            ranks[idx] = rank
        
        return ranks.astype(float)

