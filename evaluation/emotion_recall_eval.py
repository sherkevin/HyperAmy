#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HyperAmy 基于情绪相似度的评估方法

实现 EmotionRecall@K 指标，用于评估纯情绪检索的性能。
与传统的精确匹配（Exact Match）不同，EmotionRecall@K 基于情绪向量相似度。
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 情绪相似度阈值（可调整）
DEFAULT_EMOTION_SIMILARITY_THRESHOLD = 0.7  # 余弦相似度阈值
DEFAULT_HYPERBOLIC_DISTANCE_THRESHOLD = 0.5  # 双曲距离阈值（越小越相似）


@dataclass
class EmotionRecallResult:
    """情绪相似度召回结果"""
    emotion_recall_at_k: Dict[int, float]  # {1: 0.5, 5: 0.7, ...}
    emotion_similarity_scores: List[List[float]]  # 每个查询的Top-K相似度分数
    threshold: float  # 使用的相似度阈值
    total_queries: int
    hits_at_k: Dict[int, int]  # {1: 10, 5: 20, ...}


class EmotionRecallEvaluator:
    """
    基于情绪相似度的评估器
    
    用于评估HyperAmy纯情绪检索的性能，不依赖精确匹配。
    """
    
    def __init__(
        self,
        similarity_threshold: float = DEFAULT_EMOTION_SIMILARITY_THRESHOLD,
        use_hyperbolic_distance: bool = False,
        hyperbolic_distance_threshold: float = DEFAULT_HYPERBOLIC_DISTANCE_THRESHOLD
    ):
        """
        初始化评估器
        
        Args:
            similarity_threshold: 情绪相似度阈值（余弦相似度，0-1之间）
            use_hyperbolic_distance: 是否使用双曲距离（HyperAmy使用双曲空间）
            hyperbolic_distance_threshold: 双曲距离阈值（越小越相似）
        """
        self.similarity_threshold = similarity_threshold
        self.use_hyperbolic_distance = use_hyperbolic_distance
        self.hyperbolic_distance_threshold = hyperbolic_distance_threshold
        
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        # 确保向量归一化
        vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1, vec2))
    
    def hyperbolic_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算双曲距离（Poincaré球模型）
        
        使用标准的Poincaré距离公式:
        d(u,v) = arccosh(1 + 2 * ||u-v||^2 / ((1-||u||^2) * (1-||v||^2)))
        
        Args:
            vec1: 第一个向量（已归一化的情绪向量）
            vec2: 第二个向量（已归一化的情绪向量）
            
        Returns:
            双曲距离值
        """
        import math
        
        # 确保向量在单位圆盘内（模长 < 1）
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        
        # 归一化到单位圆盘
        if vec1_norm >= 1.0:
            vec1 = vec1 / (vec1_norm + 1e-8) * 0.99
            vec1_norm = 0.99
        if vec2_norm >= 1.0:
            vec2 = vec2 / (vec2_norm + 1e-8) * 0.99
            vec2_norm = 0.99
        
        diff = vec1 - vec2
        diff_norm_sq = np.dot(diff, diff)
        
        norm1_sq = np.dot(vec1, vec1)
        norm2_sq = np.dot(vec2, vec2)
        
        # 避免除零和负数
        denominator = (1 - norm1_sq + 1e-8) * (1 - norm2_sq + 1e-8)
        if denominator <= 0:
            return float('inf')
        
        # 计算Poincaré距离
        # d(u,v) = arccosh(1 + 2 * ||u-v||^2 / ((1-||u||^2) * (1-||v||^2)))
        inner = 1.0 + 2.0 * diff_norm_sq / denominator
        
        # 确保arccosh的参数 >= 1
        if inner < 1.0:
            inner = 1.0
        
        try:
            distance = math.acosh(inner)
        except ValueError:
            # 如果数值问题导致无法计算arccosh，使用近似值
            distance = np.sqrt(diff_norm_sq)
        
        return float(distance)
    
    def compute_similarity(
        self,
        gold_emotion_vector: np.ndarray,
        retrieved_emotion_vector: np.ndarray
    ) -> float:
        """
        计算情绪相似度
        
        Returns:
            相似度分数：
            - 如果使用余弦相似度：返回0-1之间的分数（越大越相似）
            - 如果使用双曲距离：返回距离值（越小越相似，需要转换为相似度）
        """
        if self.use_hyperbolic_distance:
            distance = self.hyperbolic_distance(gold_emotion_vector, retrieved_emotion_vector)
            # 将距离转换为相似度（距离越小，相似度越高）
            # 使用 exp(-distance) 或 1/(1+distance) 进行转换
            similarity = 1.0 / (1.0 + distance)
            return similarity
        else:
            return self.cosine_similarity(gold_emotion_vector, retrieved_emotion_vector)
    
    def is_similar(
        self,
        gold_emotion_vector: np.ndarray,
        retrieved_emotion_vector: np.ndarray
    ) -> Tuple[bool, float]:
        """
        判断两个情绪向量是否相似
        
        Returns:
            (is_similar, similarity_score): 是否相似和相似度分数
        """
        similarity = self.compute_similarity(gold_emotion_vector, retrieved_emotion_vector)
        
        if self.use_hyperbolic_distance:
            # 对于双曲距离，相似度越高（距离越小）表示越相似
            is_similar = similarity >= (1.0 / (1.0 + self.hyperbolic_distance_threshold))
        else:
            # 对于余弦相似度，分数越高表示越相似
            is_similar = similarity >= self.similarity_threshold
        
        return is_similar, similarity
    
    def calculate_emotion_recall_at_k(
        self,
        gold_emotion_vectors: List[np.ndarray],
        retrieved_emotion_vectors_list: List[List[np.ndarray]],
        k_list: List[int] = [1, 2, 5, 10]
    ) -> EmotionRecallResult:
        """
        计算EmotionRecall@K
        
        Args:
            gold_emotion_vectors: 每个查询的gold情绪向量列表
            retrieved_emotion_vectors_list: 每个查询的检索结果情绪向量列表（按相似度排序）
            k_list: 要计算的K值列表
        
        Returns:
            EmotionRecallResult对象
        """
        assert len(gold_emotion_vectors) == len(retrieved_emotion_vectors_list), \
            f"查询数量不匹配: {len(gold_emotion_vectors)} vs {len(retrieved_emotion_vectors_list)}"
        
        total_queries = len(gold_emotion_vectors)
        k_list = sorted(set(k_list))
        
        # 初始化结果
        hits_at_k = {k: 0 for k in k_list}
        emotion_similarity_scores = []
        
        # 对每个查询计算
        for query_idx, (gold_vec, retrieved_vecs) in enumerate(
            zip(gold_emotion_vectors, retrieved_emotion_vectors_list)
        ):
            # 计算Top-K的相似度分数（只计算一次，避免重复计算）
            max_k = k_list[-1] if k_list else len(retrieved_vecs)
            query_similarities = []
            
            # 对每个检索结果计算相似度
            for retrieved_vec in retrieved_vecs[:max_k]:
                _, similarity = self.is_similar(gold_vec, retrieved_vec)
                query_similarities.append(similarity)
            
            # 保存相似度分数
            emotion_similarity_scores.append(query_similarities)
            
            # 对每个K值检查是否有相似的结果
            for k in k_list:
                if k <= len(retrieved_vecs):
                    # 检查Top-K中是否有相似的结果
                    top_k_vecs = retrieved_vecs[:k]
                    has_similar = any(
                        self.is_similar(gold_vec, vec)[0]
                        for vec in top_k_vecs
                    )
                    if has_similar:
                        hits_at_k[k] += 1
        
        # 计算Recall@K
        emotion_recall_at_k = {
            k: hits_at_k[k] / total_queries if total_queries > 0 else 0.0
            for k in k_list
        }
        
        # 确定使用的阈值
        threshold = self.hyperbolic_distance_threshold if self.use_hyperbolic_distance else self.similarity_threshold
        
        return EmotionRecallResult(
            emotion_recall_at_k=emotion_recall_at_k,
            emotion_similarity_scores=emotion_similarity_scores,
            threshold=threshold,
            total_queries=total_queries,
            hits_at_k=hits_at_k
        )


def evaluate_hyperamy_emotion_recall(
    queries: List[str],
    gold_texts: List[str],
    retrieved_texts_list: List[List[str]],
    gold_emotion_vectors: Optional[List[np.ndarray]] = None,
    retrieved_emotion_vectors_list: Optional[List[List[np.ndarray]]] = None,
    emotion_extractor=None,
    k_list: List[int] = [1, 2, 5, 10],
    similarity_threshold: float = DEFAULT_EMOTION_SIMILARITY_THRESHOLD,
    use_hyperbolic_distance: bool = True
) -> EmotionRecallResult:
    """
    评估HyperAmy的情绪相似度召回率
    
    Args:
        queries: 查询文本列表
        gold_texts: 每个查询的gold文本列表
        retrieved_texts_list: 每个查询的检索结果文本列表
        gold_emotion_vectors: 可选的gold情绪向量列表（如果提供则跳过提取）
        retrieved_emotion_vectors_list: 可选的检索结果情绪向量列表（如果提供则跳过提取）
        emotion_extractor: 情绪提取器（Emotion类实例）
        k_list: K值列表
        similarity_threshold: 相似度阈值
        use_hyperbolic_distance: 是否使用双曲距离
    
    Returns:
        EmotionRecallResult对象
    """
    if emotion_extractor is None:
        from particle.emotion import Emotion
        emotion_extractor = Emotion()
    
    # 提取gold情绪向量
    if gold_emotion_vectors is None:
        logger.info("提取gold文本的情绪向量...")
        gold_emotion_vectors = []
        for gold_text in gold_texts:
            if gold_text:
                emotion_vec = emotion_extractor.extract(gold_text)
                gold_emotion_vectors.append(emotion_vec)
            else:
                gold_emotion_vectors.append(np.zeros(30))  # 假设30维（需要根据实际调整）
    
    # 提取检索结果的情绪向量
    if retrieved_emotion_vectors_list is None:
        logger.info("提取检索结果的情绪向量...")
        retrieved_emotion_vectors_list = []
        for retrieved_texts in retrieved_texts_list:
            query_retrieved_vecs = []
            for text in retrieved_texts:
                if text:
                    emotion_vec = emotion_extractor.extract(text)
                    query_retrieved_vecs.append(emotion_vec)
                else:
                    query_retrieved_vecs.append(np.zeros(30))  # 假设30维
            retrieved_emotion_vectors_list.append(query_retrieved_vecs)
    
    # 创建评估器
    evaluator = EmotionRecallEvaluator(
        similarity_threshold=similarity_threshold,
        use_hyperbolic_distance=use_hyperbolic_distance
    )
    
    # 计算Recall@K
    result = evaluator.calculate_emotion_recall_at_k(
        gold_emotion_vectors=gold_emotion_vectors,
        retrieved_emotion_vectors_list=retrieved_emotion_vectors_list,
        k_list=k_list
    )
    
    return result

