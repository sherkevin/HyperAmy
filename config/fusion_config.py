#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
融合策略配置常量

定义最佳配置和常用配置，方便统一管理
"""

from sentiment.fusion_strategies import FusionStrategy, NormalizationStrategy

# 最佳配置（基于网格搜索结果）
BEST_CONFIG = {
    'strategy': FusionStrategy.HARMONIC,
    'normalization': NormalizationStrategy.NONE,
    'sentiment_weight': 0.4
}

# 默认配置（已更新为最佳配置）
DEFAULT_FUSION_STRATEGY = FusionStrategy.HARMONIC
DEFAULT_NORMALIZATION_STRATEGY = NormalizationStrategy.NONE
DEFAULT_SENTIMENT_WEIGHT = 0.4

# 常用配置预设
PRESETS = {
    'best': BEST_CONFIG,
    'linear_minmax': {
        'strategy': FusionStrategy.LINEAR,
        'normalization': NormalizationStrategy.MIN_MAX,
        'sentiment_weight': 0.5
    },
    'harmonic_minmax': {
        'strategy': FusionStrategy.HARMONIC,
        'normalization': NormalizationStrategy.MIN_MAX,
        'sentiment_weight': 0.5
    },
    'rank_fusion': {
        'strategy': FusionStrategy.RANK_FUSION,
        'normalization': NormalizationStrategy.NONE,  # Rank Fusion不需要归一化
        'sentiment_weight': 0.5  # Rank Fusion不使用权重
    }
}


def get_config(preset_name: str = 'best'):
    """
    获取预设配置
    
    Args:
        preset_name: 预设名称（'best', 'linear_minmax', 'harmonic_minmax', 'rank_fusion'）
        
    Returns:
        配置字典
    """
    if preset_name in PRESETS:
        return PRESETS[preset_name].copy()
    else:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

