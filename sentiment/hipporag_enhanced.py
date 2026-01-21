"""
HippoRAG Enhanced Module

Extends HippoRAG with emotion analysis capabilities
"""
import sys
import os

# DEBUG: 模块开始导入
print("DEBUG: sentiment.hipporag_enhanced.py - Starting imports...", flush=True)

import numpy as np
print("DEBUG: sentiment.hipporag_enhanced.py - numpy imported", flush=True)

from typing import List, Optional, Tuple, Dict, Any, Union, TYPE_CHECKING
print("DEBUG: sentiment.hipporag_enhanced.py - typing imported", flush=True)

import logging
import time
print("DEBUG: sentiment.hipporag_enhanced.py - logging, time imported", flush=True)

from concurrent.futures import ThreadPoolExecutor, as_completed
print("DEBUG: sentiment.hipporag_enhanced.py - concurrent.futures imported", flush=True)

# 延迟导入：使用__import__在运行时动态导入，避免顶层导入导致死锁
print("DEBUG: sentiment.hipporag_enhanced.py - Lazy importing hipporag (will import on first use)...", flush=True)

def _lazy_import_hipporag():
    """延迟导入hipporag模块（避免导入时死锁）"""
    try:
        from hipporag.HippoRAG import HippoRAG
        from hipporag.utils.misc_utils import compute_mdhash_id
        return HippoRAG, compute_mdhash_id
    except Exception as e:
        print(f"DEBUG: sentiment.hipporag_enhanced.py - Failed to import hipporag: {e}", flush=True)
        raise

# 延迟导入：在类定义时先不导入，在__init__中再导入
_HippoRAG = None
_compute_mdhash_id = None

print("DEBUG: sentiment.hipporag_enhanced.py - Skipped heavy hipporag import (will import lazily in __init__)", flush=True)

print("DEBUG: sentiment.hipporag_enhanced.py - Importing emotion modules...", flush=True)
from .emotion_vector import EmotionExtractor
print("DEBUG: sentiment.hipporag_enhanced.py - emotion_vector imported", flush=True)

from .emotion_store import EmotionStore
print("DEBUG: sentiment.hipporag_enhanced.py - emotion_store imported", flush=True)

from .fusion_strategies import (
    FusionStrategy,
    NormalizationStrategy,
    ScoreFusion
)
print("DEBUG: sentiment.hipporag_enhanced.py - fusion_strategies imported", flush=True)

from llm.config import BETA_WARPING
print("DEBUG: sentiment.hipporag_enhanced.py - llm.config imported", flush=True)

logger = logging.getLogger(__name__)
print("DEBUG: sentiment.hipporag_enhanced.py - All imports complete!", flush=True)


# 使用元类动态处理基类，避免顶层导入
class _HippoRAGMeta(type):
    """元类：在类定义时延迟导入基类"""
    def __new__(cls, name, bases, namespace):
        # 延迟导入HippoRAG作为基类
        global _HippoRAG
        if _HippoRAG is None:
            _HippoRAG, _ = _lazy_import_hipporag()
        # 将HippoRAG作为基类
        bases = (_HippoRAG,) + bases
        return super().__new__(cls, name, bases, namespace)


class HippoRAGEnhanced(metaclass=_HippoRAGMeta):
    """
    Enhanced HippoRAG with emotion analysis capabilities
    """
    
    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 enable_sentiment: bool = False,
                 sentiment_weight: float = 0.4,  # 更新为最佳权重
                 sentiment_model_name: Optional[str] = None,
                 enable_poincare_warping: bool = False,
                 beta_warping: float = BETA_WARPING,
                 max_workers: int = 10,  # 并发处理线程数
                 fusion_strategy: Union[FusionStrategy, str] = FusionStrategy.HARMONIC,  # 更新为最佳策略
                 normalization_strategy: Union[NormalizationStrategy, str] = NormalizationStrategy.NONE,  # 更新为最佳归一化
                 **kwargs):
        """
        Initialize HippoRAGEnhanced
        
        Args:
            global_config: BaseConfig instance
            save_dir: Directory to save outputs
            llm_model_name: LLM model name
            llm_base_url: LLM base URL
            embedding_model_name: Embedding model name
            embedding_base_url: Embedding base URL
            enable_sentiment: Whether to enable sentiment analysis
            sentiment_weight: Weight for sentiment similarity (0-1)
            sentiment_model_name: Model name for sentiment extraction
            enable_poincare_warping: Whether to enable Poincaré warping for retrieval
            beta_warping: Beta parameter for Poincaré warping formula (default: 10)
            fusion_strategy: Score fusion strategy (LINEAR, HARMONIC, GEOMETRIC, RANK_FUSION)
            normalization_strategy: Score normalization strategy (MIN_MAX, Z_SCORE, L2, NONE)
            max_workers: Number of concurrent workers for emotion extraction
            **kwargs: Additional arguments passed to HippoRAG
        """
        # 确保compute_mdhash_id已导入（用于后续使用）
        global _compute_mdhash_id
        if _compute_mdhash_id is None:
            _, _compute_mdhash_id = _lazy_import_hipporag()
            print("DEBUG: sentiment.hipporag_enhanced.py - compute_mdhash_id imported (lazy)", flush=True)
        
        # Initialize base HippoRAG
        super().__init__(
            global_config=global_config,
            save_dir=save_dir,
            llm_model_name=llm_model_name,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_model_name,
            embedding_base_url=embedding_base_url,
            **kwargs
        )
        
        self.enable_sentiment = enable_sentiment
        self.sentiment_weight = sentiment_weight
        self.semantic_weight = 1.0 - sentiment_weight
        self.enable_poincare_warping = enable_poincare_warping
        self.beta_warping = beta_warping
        self.max_workers = max_workers  # 并发线程数
        
        # Document index for Path B (Global Emotion Search)
        # Maps hash_id to document text
        # IMPORTANT: Initialize as empty dict, will be populated during indexing
        self._hash_to_doc = {}
        
        # 融合策略配置
        if isinstance(fusion_strategy, str):
            try:
                self.fusion_strategy = FusionStrategy(fusion_strategy.lower())
            except ValueError:
                logger.warning(f"Unknown fusion strategy: {fusion_strategy}, using LINEAR")
                self.fusion_strategy = FusionStrategy.LINEAR
        else:
            self.fusion_strategy = fusion_strategy
        
        if isinstance(normalization_strategy, str):
            try:
                self.normalization_strategy = NormalizationStrategy(normalization_strategy.lower())
            except ValueError:
                logger.warning(f"Unknown normalization strategy: {normalization_strategy}, using MIN_MAX")
                self.normalization_strategy = NormalizationStrategy.MIN_MAX
        else:
            self.normalization_strategy = normalization_strategy
        
        # Initialize emotion extractor and store if enabled
        if self.enable_sentiment or self.enable_poincare_warping:
            # 启用缓存以加速情绪向量提取
            self.emotion_extractor = EmotionExtractor(
                model_name=sentiment_model_name or llm_model_name,
                enable_cache=True,  # 启用缓存
                cache_dir=None  # 使用默认缓存目录
            )
            
            # Create emotion store directory
            emotion_store_dir = os.path.join(self.working_dir, "chunk_emotions")
            os.makedirs(emotion_store_dir, exist_ok=True)
            # Use delayed saving (save every 50 vectors) to improve performance
            self.emotion_store = EmotionStore(store_dir=emotion_store_dir, namespace="emotion", 
                                             auto_save=True, save_interval=50)
            
            logger.info(f"Emotion analysis enabled with weight: {self.sentiment_weight}")
        else:
            self.emotion_extractor = None
            self.emotion_store = None
    
    def index(self, docs: List[str]):
        """
        Index documents with optional emotion analysis
        
        Args:
            docs: List of documents to index
        """
        # Extract and store emotion vectors if enabled
        if self.enable_sentiment:
            logger.info(f"Extracting emotion vectors for {len(docs)} documents...")
            from tqdm import tqdm
            
            # Collect documents that need emotion extraction
            docs_to_process = []
            hash_ids_to_process = []
            for doc in docs:
                hash_id = _compute_mdhash_id(doc)
                if not self.emotion_store.contains(hash_id):
                    docs_to_process.append(doc)
                    hash_ids_to_process.append(hash_id)
            
            if docs_to_process:
                logger.info(f"Processing {len(docs_to_process)} new documents (skipping {len(docs) - len(docs_to_process)} existing)...")
                logger.info(f"Using concurrent processing with {self.max_workers} workers")
                
                # Process documents with concurrent execution
                emotion_vectors_batch = []
                hash_ids_batch = []
                
                def extract_emotion(doc_hash_pair):
                    """提取单个文档的情感向量（用于并发调用）"""
                    doc, hash_id = doc_hash_pair
                    max_retries = 3
                    retry_delay = 1.0
                    
                    for attempt in range(max_retries):
                        try:
                            if not isinstance(doc, str) or len(doc.strip()) < 10:
                                logger.warning(f"Document too short or invalid: {hash_id[:8]}...")
                                return hash_id, np.zeros(30), "Document too short"
                            
                            emotion_vector = self.emotion_extractor.extract_emotion_vector(doc)
                            
                            # 验证提取的向量
                            if emotion_vector is None:
                                raise ValueError("emotion_extractor returned None")
                            if not isinstance(emotion_vector, np.ndarray):
                                emotion_vector = np.array(emotion_vector)
                            if len(emotion_vector.shape) != 1 or emotion_vector.shape[0] == 0:
                                raise ValueError(f"Invalid emotion vector shape: {emotion_vector.shape}")
                            
                            return hash_id, emotion_vector, None
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.debug(f"Retry {attempt + 1}/{max_retries} for {hash_id[:8]}...: {e}")
                                time.sleep(retry_delay * (attempt + 1))  # 指数退避
                                continue
                            else:
                                logger.warning(f"Failed to extract emotion vector for document {hash_id[:8]}... after {max_retries} attempts: {e}")
                                # 返回零向量作为fallback
                                fallback_vector = np.zeros(30)
                                return hash_id, fallback_vector, str(e)
                    
                    # 不应该到达这里，但为了安全起见
                    return hash_id, np.zeros(30), "Unknown error"
                
                # 使用ThreadPoolExecutor并发处理
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 提交所有任务
                    future_to_pair = {
                        executor.submit(extract_emotion, (doc, hash_id)): (doc, hash_id)
                        for doc, hash_id in zip(docs_to_process, hash_ids_to_process)
                    }
                    
                    # 收集结果（带进度条）
                    results_dict = {}  # 用于保持顺序
                    for future in tqdm(as_completed(future_to_pair), 
                                     total=len(future_to_pair), 
                                     desc="Extracting emotion vectors"):
                        try:
                            hash_id, emotion_vector, error = future.result()
                            results_dict[hash_id] = emotion_vector
                            if error:
                                logger.debug(f"Document {hash_id[:8]}... extracted with fallback")
                        except Exception as e:
                            doc, hash_id = future_to_pair[future]
                            logger.error(f"Failed to process document {hash_id[:8]}...: {e}")
                            results_dict[hash_id] = np.zeros(30)
                    
                    # 按照原始顺序整理结果
                    for hash_id in hash_ids_to_process:
                        if hash_id in results_dict:
                            emotion_vectors_batch.append(results_dict[hash_id])
                            hash_ids_batch.append(hash_id)
                
                # Batch save all emotion vectors
                if emotion_vectors_batch:
                    logger.info(f"Batch saving {len(emotion_vectors_batch)} emotion vectors...")
                    self.emotion_store.batch_set(hash_ids_batch, emotion_vectors_batch, force_save=True)
                    logger.info(f"✅ Saved {len(emotion_vectors_batch)} emotion vectors")
            else:
                logger.info("All documents already have emotion vectors, skipping extraction")
        
        # Build hash_id to document mapping for Path B (Global Emotion Search)
        for doc in docs:
            hash_id = _compute_mdhash_id(doc)
            self._hash_to_doc[hash_id] = doc
        
        # Call parent index method
        super().index(docs)
    
    def _global_emotion_search(self,
                               query_emotion_embedding: np.ndarray,
                               top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Perform global emotion vector search across all stored documents
        
        Args:
            query_emotion_embedding: Query emotion vector (numpy array)
            top_k: Number of top results to return
            
        Returns:
            List of tuples (hash_id, score) sorted by score (descending)
            Score is cosine similarity (1 - distance), higher is better
        """
        if self.emotion_store is None or len(self.emotion_store.hash_ids) == 0:
            logger.warning("Emotion store is empty or not initialized, returning empty results")
            return []
        
        # Normalize query vector
        query_norm = query_emotion_embedding / (np.linalg.norm(query_emotion_embedding) + 1e-8)
        
        # Compute cosine similarity for all stored vectors
        similarities = []
        hash_ids = []
        
        for hash_id in self.emotion_store.hash_ids:
            emotion_vec = self.emotion_store.get(hash_id)
            if emotion_vec is None:
                continue
            
            # Normalize document vector
            doc_vec = np.array(emotion_vec)
            
            # Handle dimension mismatch: pad or truncate to match query dimension
            if len(doc_vec) != len(query_emotion_embedding):
                if len(doc_vec) < len(query_emotion_embedding):
                    # Pad with zeros
                    padding = np.zeros(len(query_emotion_embedding) - len(doc_vec), dtype=doc_vec.dtype)
                    doc_vec = np.concatenate([doc_vec, padding])
                else:
                    # Truncate
                    doc_vec = doc_vec[:len(query_emotion_embedding)]
            
            doc_norm = doc_vec / (np.linalg.norm(doc_vec) + 1e-8)
            
            # Compute cosine similarity (dot product of normalized vectors)
            cosine_sim = float(np.dot(query_norm, doc_norm))
            
            # Convert to score (ensure positive, map from [-1, 1] to [0, 1])
            score = (cosine_sim + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
            
            similarities.append(score)
            hash_ids.append(hash_id)
        
        # Sort by score (descending) and return top_k
        sorted_pairs = sorted(zip(hash_ids, similarities), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_k]
    
    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None):
        """
        Retrieve documents with adaptive dual-path retrieval (Experiment 4)
        
        Path A (Hybrid Re-ranking): When semantic confidence is high, use HippoRAG candidates + emotion fusion
        Path B (Global Emotion Search): When semantic collapse detected, bypass HippoRAG and search full corpus
        
        Args:
            queries: List of query strings
            num_to_retrieve: Number of documents to retrieve
            gold_docs: Gold standard documents for evaluation
            
        Returns:
            Retrieval results (same format as HippoRAG.retrieve)
        """
        # If sentiment is disabled, use base retrieval
        if not self.enable_sentiment:
            return super().retrieve(queries, num_to_retrieve, gold_docs)
        
        # Semantic collapse threshold (Experiment 4: dual-path switching)
        SEMANTIC_COLLAPSE_THRESHOLD = 0.01
        
        # Get base retrieval results (retrieve more to have candidates for re-ranking)
        expanded_k = (num_to_retrieve or self.global_config.retrieval_top_k) * 2
        base_results = super().retrieve(queries, num_to_retrieve=expanded_k, gold_docs=gold_docs)
        
        # Handle evaluation results (if gold_docs provided)
        if gold_docs is not None:
            results, eval_results = base_results
        else:
            results = base_results
            eval_results = None
        
        logger.info("Enhancing retrieval with adaptive dual-path retrieval (Experiment 4)...")
        
        # Enhance each query result with adaptive dual-path retrieval
        enhanced_results = []
        for query, result in zip(queries, results):
            try:
                # 输入验证
                if not isinstance(query, str) or len(query.strip()) < 5:
                    logger.warning(f"Query too short or invalid, skipping enhancement")
                    enhanced_results.append(result)
                    continue
                
                if result is None:
                    logger.warning(f"Result is None for query, skipping enhancement")
                    enhanced_results.append(None)
                    continue
                
                # Extract emotion vector for query (with retry)
                query_emotion = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        query_emotion = self.emotion_extractor.extract_emotion_vector(query)
                        if query_emotion is not None:
                            break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.debug(f"Retry {attempt + 1}/{max_retries} for query emotion extraction: {e}")
                            time.sleep(0.5 * (attempt + 1))
                        else:
                            logger.warning(f"Failed to extract query emotion vector after {max_retries} attempts: {e}")
                            query_emotion = np.zeros(30)  # Fallback
                
                if query_emotion is None:
                    query_emotion = np.zeros(30)
                
                # ========== Step 2: Calculate semantic confidence (S_sem) ==========
                semantic_scores = result.doc_scores.copy() if result.doc_scores is not None else np.ones(len(result.docs))
                
                # 诊断：检查semantic_scores的实际值
                if len(semantic_scores) > 0:
                    # 如果所有分数都是1.0，可能是归一化问题或HippoRAG返回的默认值
                    # 对于Vibe Search任务，真实的语义分数应该较低
                    max_score = float(np.max(semantic_scores))
                    min_score = float(np.min(semantic_scores))
                    mean_score = float(np.mean(semantic_scores))
                    
                    # 如果max_score是1.0且所有分数都接近1.0，可能是归一化问题
                    # 使用mean_score或min_score作为S_sem可能更合理
                    if max_score >= 0.99 and min_score >= 0.99:
                        # 所有分数都是1.0，使用mean_score（实际上也是1.0）
                        # 但这种情况应该触发Path B，因为1.0可能是归一化后的值
                        # 对于Vibe Search，真实的语义匹配分数应该较低
                        # 如果HippoRAG返回的都是1.0，说明它没有找到真正的匹配
                        # 我们应该将其视为语义崩溃
                        S_sem = 0.0  # 强制触发Path B
                        logger.warning(f"All semantic scores are 1.0 (max={max_score:.4f}, min={min_score:.4f}, mean={mean_score:.4f}), "
                                     f"likely normalization issue. Setting S_sem=0.0 to trigger Path B.")
                    else:
                        # 正常情况：使用max_score作为S_sem
                        S_sem = max_score
                        logger.debug(f"Semantic scores: max={max_score:.4f}, min={min_score:.4f}, mean={mean_score:.4f}, S_sem={S_sem:.4f}")
                else:
                    S_sem = 0.0
                
                # ========== Step 3: Adaptive dual-path switching ==========
                if S_sem < SEMANTIC_COLLAPSE_THRESHOLD:
                    # ========== Path B: Global Emotion Search ==========
                    logger.info(f"[DUAL-PATH] ⚠️ Semantic Collapse (S_sem={S_sem:.4f}). SWITCHING to Path B: Global Emotion Search.")
                    
                    # Perform global emotion search
                    global_results = self._global_emotion_search(
                        query_emotion_embedding=query_emotion,
                        top_k=expanded_k
                    )
                    
                    # Convert hash_ids to documents using the document index
                    enhanced_docs = []
                    enhanced_scores = []
                    for hash_id, score in global_results:
                        if hash_id in self._hash_to_doc:
                            enhanced_docs.append(self._hash_to_doc[hash_id])
                            enhanced_scores.append(score)
                        else:
                            logger.debug(f"Document with hash_id {hash_id[:8]}... not found in document index, skipping")
                    
                    # If we don't have enough results, fall back to base result
                    if len(enhanced_docs) < (num_to_retrieve or self.global_config.retrieval_top_k):
                        logger.warning(f"Global emotion search returned only {len(enhanced_docs)} documents, falling back to base result")
                        enhanced_docs = result.docs[:num_to_retrieve or self.global_config.retrieval_top_k]
                        enhanced_scores = semantic_scores[:num_to_retrieve or self.global_config.retrieval_top_k]
                    else:
                        # Take top-k
                        final_k = num_to_retrieve or self.global_config.retrieval_top_k
                        enhanced_docs = enhanced_docs[:final_k]
                        enhanced_scores = np.array(enhanced_scores[:final_k])
                    
                    # Set weights for Path B (pure emotion)
                    w_emo = 1.0
                    w_sem = 0.0
                    
                else:
                    # ========== Path A: Hybrid Re-ranking ==========
                    logger.info(f"[DUAL-PATH] ✅ Semantic Signal Strong (S_sem={S_sem:.4f}). Using Path A: Hybrid Re-ranking.")
                    
                    # Get emotion vectors for retrieved documents
                    doc_emotions = []
                    doc_hash_ids = []
                    for doc in result.docs:
                        hash_id = _compute_mdhash_id(doc)
                        doc_hash_ids.append(hash_id)
                        emotion_vec = self.emotion_store.get(hash_id)
                        if emotion_vec is None:
                            # If emotion vector not found, extract it
                            logger.warning(f"Emotion vector not found for doc, extracting...")
                            try:
                                emotion_vec = self.emotion_extractor.extract_emotion_vector(doc)
                                self.emotion_store.set(hash_id, emotion_vec)
                            except Exception as e:
                                logger.warning(f"Failed to extract emotion vector: {e}")
                                emotion_vec = np.zeros(30)  # Fallback to zero vector
                        doc_emotions.append(emotion_vec)
                    
                    # Calculate emotion similarities (cosine similarity)
                    doc_emotions = np.array(doc_emotions)
                    # Normalize vectors for cosine similarity
                    query_emotion_norm = query_emotion / (np.linalg.norm(query_emotion) + 1e-8)
                    doc_emotions_norm = doc_emotions / (np.linalg.norm(doc_emotions, axis=1, keepdims=True) + 1e-8)
                    emotion_similarities = np.dot(doc_emotions_norm, query_emotion_norm)
                    
                    # Normalize emotion similarities to [0, 1] range
                    emotion_similarities = (emotion_similarities + 1) / 2  # Cosine similarity is [-1, 1], map to [0, 1]
                    
                    # 使用融合策略融合分数（解除权重限制）
                    combined_scores, fusion_info = ScoreFusion.fuse(
                        semantic_scores=semantic_scores,
                        emotion_scores=emotion_similarities,
                        sentiment_weight=self.sentiment_weight,
                        strategy=self.fusion_strategy,
                        semantic_weight=self.semantic_weight,
                        normalization_strategy=self.normalization_strategy
                    )
                    
                    logger.debug(f"Fusion strategy: {fusion_info['strategy']}, "
                               f"semantic_mean: {fusion_info['semantic_mean']:.4f}, "
                               f"emotion_mean: {fusion_info['emotion_mean']:.4f}, "
                               f"fused_mean: {fusion_info['fused_mean']:.4f}")
                    
                    # Re-rank documents based on combined scores
                    sorted_indices = np.argsort(combined_scores)[::-1]  # Descending order
                    
                    # Get top-k documents
                    final_k = num_to_retrieve or self.global_config.retrieval_top_k
                    top_indices = sorted_indices[:final_k]
                    
                    # Create enhanced result
                    enhanced_docs = [result.docs[i] for i in top_indices]
                    enhanced_scores = combined_scores[top_indices]
                    
                    # Extract weights from fusion_info
                    w_emo = fusion_info.get('sentiment_weight', self.sentiment_weight)
                    w_sem = fusion_info.get('semantic_weight', self.semantic_weight)
                
                # Create new QuerySolution with enhanced results
                from hipporag.utils.misc_utils import QuerySolution
                enhanced_result = QuerySolution(
                    question=result.question,
                    docs=enhanced_docs,
                    doc_scores=enhanced_scores,
                    answer=result.answer,
                    gold_answers=result.gold_answers,
                    gold_docs=result.gold_docs
                )
                
                enhanced_results.append(enhanced_result)
                
                logger.debug(f"Query: {query[:50]}... | Enhanced {len(result.docs)} docs -> {len(enhanced_docs)} docs | w_emo={w_emo:.4f}, w_sem={w_sem:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to enhance retrieval for query '{query}': {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fallback to base result
                enhanced_results.append(result)
        
        logger.info(f"Emotion-enhanced retrieval completed for {len(queries)} queries")
        
        # Return results with evaluation if provided
        if eval_results is not None:
            return enhanced_results, eval_results
        else:
            return enhanced_results
    
    def rag_qa(self,
               queries: List[str],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None):
        """
        RAG QA with optional emotion awareness
        
        Args:
            queries: List of query strings
            gold_docs: Gold standard documents
            gold_answers: Gold standard answers
            
        Returns:
            QA results (same format as HippoRAG.rag_qa)
        """
        # Use base RAG QA (emotion enhancement in retrieval is handled in retrieve method)
        return super().rag_qa(queries, gold_docs, gold_answers)
    
    def retrieve_with_poincare_warping(self,
                                       query_vec: np.ndarray,
                                       doc_vecs: np.ndarray,
                                       doc_masses: np.ndarray,
                                       doc_texts: List[str],
                                       num_to_retrieve: int = 3,
                                       beta: Optional[float] = None) -> Tuple[List[str], np.ndarray]:
        """
        使用庞加莱畸变公式进行检索
        
        公式: warped_dist = cosine_dist / (1 + beta * chunk.mass)
        
        Args:
            query_vec: 查询向量 (1D numpy array)
            doc_vecs: 文档向量矩阵 (2D numpy array, shape: [n_docs, dim])
            doc_masses: 文档质量数组 (1D numpy array, shape: [n_docs])
            doc_texts: 文档文本列表
            num_to_retrieve: 要检索的文档数量
            beta: 畸变参数（如果为 None，使用 self.beta_warping）
            
        Returns:
            Tuple[List[str], np.ndarray]: (检索到的文档文本列表, 对应的分数数组)
        """
        if beta is None:
            beta = self.beta_warping
        
        # 计算余弦距离（1 - 余弦相似度）
        # 归一化向量
        query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        doc_vecs_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
        
        # 计算余弦相似度
        cosine_similarities = np.dot(doc_vecs_norm, query_vec_norm)
        
        # 转换为距离（距离 = 1 - 相似度，但相似度越高越好，所以用 1 - 相似度作为距离）
        cosine_distances = 1.0 - cosine_similarities
        
        # 应用庞加莱畸变公式: warped_dist = dist / (1 + beta * mass)
        warped_distances = cosine_distances / (1.0 + beta * doc_masses)
        
        # 排序（距离越小越好）
        sorted_indices = np.argsort(warped_distances)
        
        # 获取 top-k
        top_indices = sorted_indices[:num_to_retrieve]
        
        # 返回文档和分数（使用相似度作为分数，越高越好）
        retrieved_docs = [doc_texts[i] for i in top_indices]
        retrieved_scores = cosine_similarities[top_indices]  # 使用原始相似度作为分数
        
        return retrieved_docs, retrieved_scores

