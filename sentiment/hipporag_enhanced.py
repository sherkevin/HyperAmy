"""
增强版 HippoRAG，添加情感分析功能

这是一个包装类，在原有 HippoRAG 基础上添加情感分析能力。
"""
import os
from typing import List, Optional

from hipporag.HippoRAG import HippoRAG as BaseHippoRAG
from hipporag.utils.misc_utils import QuerySolution
from hipporag.utils.logging_utils import get_logger

from .emotion_vector import EmotionExtractor
from .emotion_store import EmotionStore
from .emotion_vector import cosine_similarity

logger = get_logger(__name__)


class HippoRAGEnhanced(BaseHippoRAG):
    """
    增强版 HippoRAG，添加情感分析功能
    
    在原有 HippoRAG 基础上添加：
    1. 情感向量提取和存储
    2. 情感增强的检索（结合语义和情感相似度）
    """
    
    def __init__(self,
                 enable_emotion=True,
                 emotion_weight=0.3,
                 emotion_model_name="GLM-4-Flash",
                 **kwargs):
        """
        初始化增强版 HippoRAG
        
        Args:
            enable_emotion: 是否启用情感分析
            emotion_weight: 情感相似度权重（0-1），剩余权重用于语义相似度
            emotion_model_name: 情感分析使用的模型名称
            **kwargs: 传递给基类的其他参数
        """
        # 初始化基类
        super().__init__(**kwargs)
        
        self.enable_emotion = enable_emotion
        self.emotion_weight = emotion_weight
        self.semantic_weight = 1.0 - emotion_weight
        
        if self.enable_emotion:
            # 初始化情感提取器
            try:
                # 尝试从配置获取 API 信息
                api_key = kwargs.get('llm_base_url')  # 临时使用，实际应该从配置读取
                api_base_url = kwargs.get('llm_base_url')
                
                self.emotion_extractor = EmotionExtractor(
                    api_key=None,  # 从环境变量读取
                    api_base_url=api_base_url,
                    model_name=emotion_model_name
                )
                
                # 初始化情感存储
                self.entity_emotion_store = EmotionStore(
                    self.emotion_extractor,
                    os.path.join(self.working_dir, "entity_emotions"),
                    self.global_config.embedding_batch_size,
                    'entity_emotion'
                )
                
                self.chunk_emotion_store = EmotionStore(
                    self.emotion_extractor,
                    os.path.join(self.working_dir, "chunk_emotions"),
                    self.global_config.embedding_batch_size,
                    'chunk_emotion'
                )
                
                logger.info(f"Emotion analysis enabled (weight: {self.emotion_weight})")
            except Exception as e:
                logger.warning(f"Failed to initialize emotion analysis: {e}. Disabling emotion features.")
                self.enable_emotion = False
        else:
            logger.info("Emotion analysis disabled")
    
    def index(self, docs: List[str]):
        """
        索引文档，添加情感向量提取
        
        Args:
            docs: 文档列表
        """
        # 调用基类的索引方法
        super().index(docs)
        
        if self.enable_emotion:
            logger.info("Extracting emotion vectors for entities and chunks")
            
            # 提取实体的情感向量
            entity_nodes = list(self.entity_embedding_store.get_all_texts())
            if entity_nodes:
                logger.info(f"Extracting emotion vectors for {len(entity_nodes)} entities")
                self.entity_emotion_store.insert_strings(list(entity_nodes))
            
            # 提取文档块的情感向量
            chunk_texts = [self.chunk_embedding_store.get_row(hash_id)["content"] 
                          for hash_id in self.chunk_embedding_store.get_all_ids()]
            if chunk_texts:
                logger.info(f"Extracting emotion vectors for {len(chunk_texts)} chunks")
                self.chunk_emotion_store.insert_strings(chunk_texts)
    
    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: Optional[int] = None,
                 gold_docs: Optional[List[List[str]]] = None):
        """
        检索文档，可选地使用情感增强
        
        Args:
            queries: 查询列表
            num_to_retrieve: 检索数量
            gold_docs: 标准文档（用于评估）
        
        Returns:
            检索结果
        """
        if not self.enable_emotion:
            # 如果未启用情感分析，直接使用基类方法
            return super().retrieve(queries, num_to_retrieve, gold_docs)
        
        # 使用情感增强的检索
        # 这里先使用基类方法，然后在结果中结合情感相似度
        results = super().retrieve(queries, num_to_retrieve, gold_docs)
        
        # 如果有评估结果，需要特殊处理
        if isinstance(results, tuple):
            retrieval_results, eval_results = results
        else:
            retrieval_results = results
            eval_results = None
        
        # 对每个查询，结合情感相似度重新排序
        enhanced_results = []
        for query_solution in retrieval_results:
            query = query_solution.question
            docs = query_solution.docs
            doc_scores = query_solution.doc_scores
            
            try:
                # 提取查询的情感向量
                query_emotion_vector, _ = self.emotion_extractor.extract_emotion_vector(query)
                
                # 计算每个文档的情感相似度
                emotion_scores = []
                for doc in docs:
                    # 获取文档的情感向量
                    # 注意：chunk_emotion_store 使用 'chunk_emotion-' 前缀
                    chunk_hash_id = self.chunk_embedding_store.get_hash_id(doc)
                    if chunk_hash_id:
                        # 尝试从 emotion store 获取（使用相同的 hash_id）
                        try:
                            doc_emotion_vector = self.chunk_emotion_store.get_emotion_vector(chunk_hash_id)
                            emotion_sim = cosine_similarity(query_emotion_vector, doc_emotion_vector)
                        except (KeyError, AttributeError):
                            # 如果文档没有情感向量，使用默认值
                            emotion_sim = 0.5
                    else:
                        # 如果文档没有情感向量，使用默认值
                        emotion_sim = 0.5
                    emotion_scores.append(emotion_sim)
                
                # 归一化语义分数和情感分数
                import numpy as np
                semantic_scores_norm = np.array(doc_scores)
                if semantic_scores_norm.max() > semantic_scores_norm.min():
                    semantic_scores_norm = (semantic_scores_norm - semantic_scores_norm.min()) / (
                        semantic_scores_norm.max() - semantic_scores_norm.min())
                else:
                    semantic_scores_norm = np.ones_like(semantic_scores_norm)
                
                emotion_scores_norm = np.array(emotion_scores)
                if emotion_scores_norm.max() > emotion_scores_norm.min():
                    emotion_scores_norm = (emotion_scores_norm - emotion_scores_norm.min()) / (
                        emotion_scores_norm.max() - emotion_scores_norm.min())
                else:
                    emotion_scores_norm = np.ones_like(emotion_scores_norm)
                
                # 结合语义和情感分数
                combined_scores = (self.semantic_weight * semantic_scores_norm + 
                                 self.emotion_weight * emotion_scores_norm)
                
                # 重新排序
                sorted_indices = np.argsort(combined_scores)[::-1]
                enhanced_docs = [docs[i] for i in sorted_indices]
                enhanced_scores = [float(combined_scores[i]) for i in sorted_indices]
                
                enhanced_solution = QuerySolution(
                    question=query,
                    docs=enhanced_docs,
                    doc_scores=enhanced_scores
                )
                enhanced_results.append(enhanced_solution)
                
            except Exception as e:
                logger.warning(f"Failed to enhance retrieval with emotion for query '{query[:50]}...': {e}")
                # 如果失败，使用原始结果
                enhanced_results.append(query_solution)
        
        if eval_results is not None:
            return enhanced_results, eval_results
        else:
            return enhanced_results

