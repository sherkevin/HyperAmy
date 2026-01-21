"""
HippoRAG Enhanced Module - Optimized Version

优化版本：使用并发处理Emotion提取
"""
import os
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.misc_utils import compute_mdhash_id
from .emotion_vector import EmotionExtractor
from .emotion_store import EmotionStore
from llm.config import BETA_WARPING

logger = logging.getLogger(__name__)


class HippoRAGEnhancedOptimized(HippoRAG):
    """
    Enhanced HippoRAG with emotion analysis capabilities - Optimized with concurrent processing
    """
    
    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 enable_sentiment: bool = False,
                 sentiment_weight: float = 0.3,
                 sentiment_model_name: Optional[str] = None,
                 enable_poincare_warping: bool = False,
                 beta_warping: float = BETA_WARPING,
                 max_workers: int = 10,  # 并发线程数
                 **kwargs):
        """
        Initialize HippoRAGEnhancedOptimized
        
        Args:
            max_workers: 并发处理线程数（默认10）
            其他参数同HippoRAGEnhanced
        """
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
        
        # Initialize emotion extractor and store if enabled
        if self.enable_sentiment or self.enable_poincare_warping:
            self.emotion_extractor = EmotionExtractor(model_name=sentiment_model_name or llm_model_name)
            
            # Create emotion store directory
            emotion_store_dir = os.path.join(self.working_dir, "chunk_emotions")
            os.makedirs(emotion_store_dir, exist_ok=True)
            # Use delayed saving (save every 50 vectors) to improve performance
            self.emotion_store = EmotionStore(store_dir=emotion_store_dir, namespace="emotion", 
                                             auto_save=True, save_interval=50)
            
            logger.info(f"Emotion analysis enabled with weight: {self.sentiment_weight}")
            logger.info(f"Concurrent processing enabled with {self.max_workers} workers")
        else:
            self.emotion_extractor = None
            self.emotion_store = None
    
    def index(self, docs: List[str]):
        """
        Index documents with optional emotion analysis - Optimized with concurrent processing
        
        Args:
            docs: List of documents to index
        """
        # Extract and store emotion vectors if enabled
        if self.enable_sentiment:
            logger.info(f"Extracting emotion vectors for {len(docs)} documents...")
            
            # Collect documents that need emotion extraction
            docs_to_process = []
            hash_ids_to_process = []
            for doc in docs:
                hash_id = compute_mdhash_id(doc)
                if not self.emotion_store.contains(hash_id):
                    docs_to_process.append(doc)
                    hash_ids_to_process.append(hash_id)
            
            if docs_to_process:
                logger.info(f"Processing {len(docs_to_process)} new documents (skipping {len(docs) - len(docs_to_process)} existing)...")
                
                # 并发处理文档
                emotion_vectors_batch = []
                hash_ids_batch = []
                
                def extract_emotion(doc_hash_pair):
                    """提取单个文档的情感向量"""
                    doc, hash_id = doc_hash_pair
                    try:
                        emotion_vector = self.emotion_extractor.extract_emotion_vector(doc)
                        return hash_id, emotion_vector, None
                    except Exception as e:
                        logger.warning(f"Failed to extract emotion vector for document: {e}")
                        return hash_id, np.zeros(30), str(e)  # 返回零向量作为fallback
                
                # 使用ThreadPoolExecutor并发处理
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 提交所有任务
                    future_to_pair = {
                        executor.submit(extract_emotion, (doc, hash_id)): (doc, hash_id)
                        for doc, hash_id in zip(docs_to_process, hash_ids_to_process)
                    }
                    
                    # 收集结果（带进度条）
                    for future in tqdm(as_completed(future_to_pair), 
                                     total=len(future_to_pair), 
                                     desc="Extracting emotion vectors"):
                        try:
                            hash_id, emotion_vector, error = future.result()
                            emotion_vectors_batch.append(emotion_vector)
                            hash_ids_batch.append(hash_id)
                            if error:
                                logger.debug(f"Document {hash_id[:8]}... extracted with fallback")
                        except Exception as e:
                            doc, hash_id = future_to_pair[future]
                            logger.error(f"Failed to process document {hash_id[:8]}...: {e}")
                            # 添加零向量作为fallback
                            emotion_vectors_batch.append(np.zeros(30))
                            hash_ids_batch.append(hash_id)
                
                # Batch save all emotion vectors
                if emotion_vectors_batch:
                    logger.info(f"Batch saving {len(emotion_vectors_batch)} emotion vectors...")
                    self.emotion_store.batch_set(hash_ids_batch, emotion_vectors_batch, force_save=True)
                    logger.info(f"✅ Saved {len(emotion_vectors_batch)} emotion vectors")
            else:
                logger.info("All documents already have emotion vectors, skipping extraction")
        
        # Call parent index method
        super().index(docs)
    
    # retrieve方法保持不变（继承自HippoRAGEnhanced）
    # 这里省略，实际使用时可以导入原类的方法或继承

