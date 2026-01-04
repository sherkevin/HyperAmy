# -*- coding: utf-8 -*-
"""
GoT 实验数据预处理模块

功能：
1. 数据获取（支持本地文件或 HuggingFace）
2. 语义分块（SemanticChunker）
3. 杏仁核特征注入（向量、情感、惊奇度、质量）
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入配置
from llm.config import CHUNK_SIZE, CHUNK_OVERLAP, MASS_THRESHOLD

# 导入模型（延迟加载以避免不必要的依赖）
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Install with: pip install transformers torch")


class SemanticChunker:
    """
    语义分块器
    
    按段落分割，合并到约300词，长段落按句子分割，50词重叠
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        初始化分块器
        
        Args:
            chunk_size: 目标块大小（词数）
            chunk_overlap: 重叠大小（词数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _count_words(self, text: str) -> int:
        """计算文本词数（简单实现）"""
        return len(text.split())
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        # 简单的句子分割（按句号、问号、感叹号）
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str) -> List[Dict[str, any]]:
        """
        对文本进行语义分块
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict]: 分块列表，每个块包含 {'text': str, 'start_idx': int, 'end_idx': int}
        """
        chunks = []
        
        # 按段落分割（双换行符）
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果当前块加上新段落不超过限制，直接合并
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            if self._count_words(test_chunk) <= self.chunk_size:
                if not current_chunk:
                    current_start = text.find(para)
                current_chunk = test_chunk
            else:
                # 需要分割
                if current_chunk:
                    # 保存当前块
                    chunks.append({
                        'text': current_chunk,
                        'start_idx': current_start,
                        'end_idx': current_start + len(current_chunk)
                    })
                
                # 处理新段落
                if self._count_words(para) <= self.chunk_size:
                    # 段落本身足够小
                    current_chunk = para
                    current_start = text.find(para)
                else:
                    # 段落太大，按句子分割
                    sentences = self._split_into_sentences(para)
                    current_chunk = ""
                    current_start = text.find(para)
                    
                    for sentence in sentences:
                        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                        if self._count_words(test_chunk) <= self.chunk_size:
                            current_chunk = test_chunk
                        else:
                            # 保存当前块
                            if current_chunk:
                                chunks.append({
                                    'text': current_chunk,
                                    'start_idx': current_start,
                                    'end_idx': current_start + len(current_chunk)
                                })
                            
                            # 开始新块（带重叠）
                            if chunks:
                                # 从上一个块的末尾提取重叠部分
                                last_chunk = chunks[-1]['text']
                                overlap_words = last_chunk.split()[-self.chunk_overlap:]
                                overlap_text = " ".join(overlap_words)
                                current_chunk = overlap_text + " " + sentence
                                current_start = chunks[-1]['end_idx'] - len(overlap_text)
                            else:
                                current_chunk = sentence
                                current_start = text.find(sentence)
        
        # 保存最后一个块
        if current_chunk:
            chunks.append({
                'text': current_chunk,
                'start_idx': current_start,
                'end_idx': current_start + len(current_chunk)
            })
        
        return chunks


class AmygdalaFeatureInjector:
    """
    杏仁核特征注入器
    
    计算：
    - 向量 z: all-MiniLM-L6-v2 embedding (384维)
    - 情感分数: roberta-base-go_emotions 负面情感检测
    - 惊奇度分数: gpt2 PPL (困惑度)
    - 质量 m: 0.7 * emotion_score + 0.3 * surprisal_score
    """
    
    def __init__(self):
        """初始化特征注入器"""
        self.embedding_model = None
        self.emotion_model = None
        self.surprisal_tokenizer = None
        self.surprisal_model = None
        self._load_models()
    
    def _load_models(self):
        """延迟加载模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers torch")
        
        logger.info("Loading embedding model: all-MiniLM-L6-v2...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Loading emotion model: SamLowe/roberta-base-go_emotions...")
        self.emotion_model = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("Loading surprisal model: gpt2...")
        self.surprisal_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.surprisal_model = AutoModelForCausalLM.from_pretrained('gpt2')
        if torch.cuda.is_available():
            self.surprisal_model = self.surprisal_model.cuda()
        self.surprisal_model.eval()
        
        # 设置 pad_token
        if self.surprisal_tokenizer.pad_token is None:
            self.surprisal_tokenizer.pad_token = self.surprisal_tokenizer.eos_token
    
    def compute_vector(self, text: str) -> np.ndarray:
        """
        计算文本的 embedding 向量
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 384维向量
        """
        if self.embedding_model is None:
            self._load_models()
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def compute_emotion_score(self, text: str) -> float:
        """
        计算负面情感分数
        
        Args:
            text: 输入文本
            
        Returns:
            float: 负面情感分数 [0, 1]
        """
        if self.emotion_model is None:
            self._load_models()
        
        # 负面情感标签
        negative_emotions = ['fear', 'surprise', 'anger', 'disgust', 'sadness', 'annoyance']
        
        # 如果文本太长，截断（模型有长度限制）
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            results = self.emotion_model(text, top_k=None)
            
            # 计算负面情感的总分数
            negative_score = 0.0
            for result in results:
                if result['label'].lower() in negative_emotions:
                    negative_score += result['score']
            
            # 归一化到 [0, 1]
            return min(negative_score, 1.0)
        except Exception as e:
            logger.warning(f"Failed to compute emotion score: {e}")
            return 0.0
    
    def compute_surprisal_score(self, text: str) -> float:
        """
        计算惊奇度分数（基于 PPL）
        
        Args:
            text: 输入文本
            
        Returns:
            float: 归一化的惊奇度分数 [0, 1]
        """
        if self.surprisal_model is None:
            self._load_models()
        
        try:
            # Tokenize
            inputs = self.surprisal_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 计算 PPL
            with torch.no_grad():
                outputs = self.surprisal_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                ppl = torch.exp(loss).item()
            
            # 归一化 PPL 到 [0, 1]
            # PPL 通常在 1-1000 之间，我们使用 log 归一化
            log_ppl = np.log(ppl + 1)
            max_log_ppl = np.log(1000 + 1)  # 假设最大 PPL 为 1000
            surprisal_score = min(log_ppl / max_log_ppl, 1.0)
            
            return surprisal_score
        except Exception as e:
            logger.warning(f"Failed to compute surprisal score: {e}")
            return 0.0
    
    def compute_mass(self, emotion_score: float, surprisal_score: float) -> float:
        """
        计算质量 m
        
        Args:
            emotion_score: 情感分数
            surprisal_score: 惊奇度分数
            
        Returns:
            float: 质量 m [0, 1]
        """
        mass = 0.7 * emotion_score + 0.3 * surprisal_score
        return np.clip(mass, 0.0, 1.0)
    
    def inject_features(self, chunks: List[Dict]) -> List[Dict]:
        """
        为所有块注入特征
        
        Args:
            chunks: 分块列表，每个块包含 {'text': str, ...}
            
        Returns:
            List[Dict]: 增强后的块列表，包含所有特征
        """
        enhanced_chunks = []
        
        logger.info(f"Computing features for {len(chunks)} chunks...")
        
        for i, chunk in enumerate(tqdm(chunks, desc="Injecting features")):
            text = chunk['text']
            
            # 计算特征
            vector = self.compute_vector(text)
            emotion_score = self.compute_emotion_score(text)
            surprisal_score = self.compute_surprisal_score(text)
            mass = self.compute_mass(emotion_score, surprisal_score)
            
            enhanced_chunk = {
                'chunk_id': i,
                'text': text,
                'vector': vector.tolist(),  # 转换为列表以便 JSON 序列化
                'emotion_score': float(emotion_score),
                'surprisal_score': float(surprisal_score),
                'mass': float(mass)
            }
            
            # 保留原始字段
            if 'start_idx' in chunk:
                enhanced_chunk['start_idx'] = chunk['start_idx']
            if 'end_idx' in chunk:
                enhanced_chunk['end_idx'] = chunk['end_idx']
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks


def load_text_from_file(file_path: str) -> str:
    """
    从本地文件加载文本
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 文本内容
    """
    # 尝试多种编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    # 如果都失败，使用错误处理
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_text_from_huggingface(dataset_name: str = "bookcorpus", split: str = "train") -> str:
    """
    从 HuggingFace 加载文本数据
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割
        
    Returns:
        str: 合并后的文本内容
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets required. Install with: pip install datasets")
    
    logger.info(f"Loading dataset {dataset_name} from HuggingFace...")
    dataset = load_dataset(dataset_name, split=split)
    
    # 合并所有文本
    texts = []
    for item in dataset:
        if 'text' in item:
            texts.append(item['text'])
        elif 'content' in item:
            texts.append(item['content'])
        else:
            # 尝试找到第一个字符串字段
            for key, value in item.items():
                if isinstance(value, str):
                    texts.append(value)
                    break
    
    return "\n\n".join(texts)


def prepare_data(
    input_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    output_path: str = "data/processed/got_amygdala.jsonl"
):
    """
    完整的数据预处理流程
    
    Args:
        input_path: 本地文件路径（如果提供）
        dataset_name: HuggingFace 数据集名称（如果提供）
        output_path: 输出文件路径
    """
    # 1. 加载文本
    if input_path:
        logger.info(f"Loading text from local file: {input_path}")
        text = load_text_from_file(input_path)
    elif dataset_name:
        logger.info(f"Loading text from HuggingFace: {dataset_name}")
        text = load_text_from_huggingface(dataset_name)
    else:
        # 尝试从默认位置加载
        default_path = "data/books/got_sample.txt"
        if os.path.exists(default_path):
            logger.info(f"Loading text from default path: {default_path}")
            text = load_text_from_file(default_path)
        else:
            raise ValueError("Either input_path or dataset_name must be provided, or place a file at data/books/got_sample.txt")
    
    logger.info(f"Loaded text with {len(text)} characters")
    
    # 2. 语义分块
    logger.info("Chunking text...")
    chunker = SemanticChunker()
    chunks = chunker.chunk(text)
    logger.info(f"Created {len(chunks)} chunks")
    
    # 3. 特征注入
    logger.info("Injecting amygdala features...")
    injector = AmygdalaFeatureInjector()
    enhanced_chunks = injector.inject_features(chunks)
    
    # 4. 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in enhanced_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(enhanced_chunks)} enhanced chunks to {output_path}")
    
    # 5. 统计信息
    masses = [chunk['mass'] for chunk in enhanced_chunks]
    high_quality_count = sum(1 for m in masses if m > MASS_THRESHOLD)
    
    logger.info(f"Statistics:")
    logger.info(f"  Total chunks: {len(enhanced_chunks)}")
    logger.info(f"  High quality chunks (mass > {MASS_THRESHOLD}): {high_quality_count}")
    logger.info(f"  Average mass: {np.mean(masses):.3f}")
    logger.info(f"  Max mass: {np.max(masses):.3f}")
    logger.info(f"  Min mass: {np.min(masses):.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GoT 实验数据预处理")
    parser.add_argument("--input", type=str, help="本地文件路径")
    parser.add_argument("--dataset", type=str, help="HuggingFace 数据集名称")
    parser.add_argument("--output", type=str, default="data/processed/got_amygdala.jsonl", help="输出文件路径")
    
    args = parser.parse_args()
    
    prepare_data(
        input_path=args.input,
        dataset_name=args.dataset,
        output_path=args.output
    )

