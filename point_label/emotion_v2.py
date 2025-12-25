"""
Emotion V2 类

整合实体抽取、情感描述生成和情绪嵌入的完整流程：
1. 调用 Entity 抽取实体
2. 调用 Sentence 对实体生成情绪描述句子
3. 对每个句子进行情绪嵌入得到嵌入向量
4. 输出为一个列表，每个 node 为一个结构体
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import requests
import os

from utils.entitiy import Entity
from utils.sentence import Sentence
from llm.config import API_KEY, API_URL_EMBEDDINGS, DEFAULT_EMBEDDING_MODEL
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EmotionNode:
    """
    情绪节点结构体
    
    包含：
    - entity_id: 实体 ID（唯一标识）
    - entity: 实体名称
    - emotion_vector: 情绪嵌入向量（高维稠密向量）
    - text_id: 原文本 ID（用于映射关系）
    """
    entity_id: str
    entity: str
    emotion_vector: np.ndarray
    text_id: str


class EmotionV2:
    """
    Emotion V2 类
    
    完整流程：
    1. 从文本中抽取实体
    2. 为每个实体生成情感视角描述
    3. 对每个情感描述进行嵌入，得到情绪向量
    4. 返回 EmotionNode 列表
    """
    
    def __init__(
        self,
        model_name=None,
        embedding_model_name=None,
        entity_extractor=None,
        sentence_processor=None
    ):
        """
        初始化 EmotionV2 类
        
        Args:
            model_name: LLM 模型名称，用于 Sentence 类
            embedding_model_name: 嵌入模型名称，用于生成情绪嵌入向量
            entity_extractor: Entity 实例（可选），如果为 None 则自动创建
            sentence_processor: Sentence 实例（可选），如果为 None 则自动创建
        """
        from llm.config import DEFAULT_MODEL
        
        self.model_name = model_name or DEFAULT_MODEL
        self.embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL
        
        # 初始化组件
        if entity_extractor is None:
            self.entity_extractor = Entity(model_name=self.model_name)
        else:
            self.entity_extractor = entity_extractor
        
        if sentence_processor is None:
            self.sentence_processor = Sentence(model_name=self.model_name)
        else:
            self.sentence_processor = sentence_processor
        
        logger.info(
            f"EmotionV2 initialized with model: {self.model_name}, "
            f"embedding_model: {self.embedding_model_name}"
        )
    
    def _get_emotion_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的情绪嵌入向量
        
        使用 embedding API 将文本转换为高维稠密向量
        
        Args:
            text: 输入文本（情感描述句子）
        
        Returns:
            numpy.ndarray: 情绪嵌入向量
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # API 支持字符串或字符串列表，这里使用列表格式以保持一致性
        payload = {
            "model": self.embedding_model_name,
            "input": [text],  # 使用列表格式
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(API_URL_EMBEDDINGS, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            # 提取 embedding 向量
            if isinstance(result.get("data"), list) and len(result["data"]) > 0:
                embedding = np.array(result["data"][0]["embedding"])
            else:
                raise ValueError(f"Unexpected API response format: {result}")
            
            logger.debug(f"Generated embedding vector of shape {embedding.shape} for text: {text[:50]}...")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get emotion embedding for text '{text[:50]}...': {e}")
            raise
    
    def process(
        self,
        text: str,
        text_id: str,
        entities: Optional[List[str]] = None
    ) -> List[EmotionNode]:
        """
        处理文本，生成情绪节点列表
        
        完整流程：
        1. 抽取实体（如果未提供）
        2. 为每个实体生成情感描述
        3. 对每个情感描述进行嵌入
        4. 返回 EmotionNode 列表
        
        Args:
            text: 原始文本
            text_id: 原文本 ID（用于映射关系）
            entities: 实体列表（可选），如果为 None 则自动抽取
        
        Returns:
            List[EmotionNode]: 情绪节点列表
        """
        # Step 1: 抽取实体（如果未提供）
        if entities is None:
            try:
                entities = self.entity_extractor.extract_entities(text)
                logger.info(f"Extracted {len(entities)} entities from text: {entities}")
            except Exception as e:
                logger.error(f"Failed to extract entities: {e}")
                return []  # 如果抽取失败，返回空列表
        
        if not entities:
            logger.warning(f"No entities found in text (text_id: {text_id})")
            return []
        
        # Step 2: 为每个实体生成情感描述
        try:
            affective_descriptions = self.sentence_processor.generate_affective_descriptions(
                sentence=text,
                entities=entities
            )
            logger.info(
                f"Generated {len([d for d in affective_descriptions.values() if d])} "
                f"affective descriptions for {len(entities)} entities"
            )
        except Exception as e:
            logger.error(f"Failed to generate affective descriptions: {e}")
            return []
        
        # Step 3: 对每个情感描述进行嵌入，生成 EmotionNode 列表
        nodes = []
        for idx, entity in enumerate(entities):
            description = affective_descriptions.get(entity, "")
            
            if not description:
                logger.warning(f"Empty affective description for entity '{entity}', skipping...")
                continue
            
            try:
                # 获取情绪嵌入向量
                emotion_vector = self._get_emotion_embedding(description)
                
                # 生成实体 ID（格式：text_id_entity_idx）
                entity_id = f"{text_id}_entity_{idx}"
                
                # 创建 EmotionNode
                node = EmotionNode(
                    entity_id=entity_id,
                    entity=entity,
                    emotion_vector=emotion_vector,
                    text_id=text_id
                )
                
                nodes.append(node)
                
                logger.debug(
                    f"Created node: entity_id={entity_id}, entity={entity}, "
                    f"vector_shape={emotion_vector.shape}, text_id={text_id}"
                )
                
            except Exception as e:
                logger.error(f"Failed to process entity '{entity}': {e}")
                continue  # 跳过失败的实体，继续处理其他实体
        
        logger.info(
            f"Successfully processed {len(nodes)}/{len(entities)} entities "
            f"for text_id: {text_id}"
        )
        
        return nodes
    
    def batch_process(
        self,
        texts: List[str],
        text_ids: Optional[List[str]] = None
    ) -> List[EmotionNode]:
        """
        批量处理多个文本
        
        Args:
            texts: 原始文本列表
            text_ids: 文本 ID 列表（可选），如果为 None 则自动生成
        
        Returns:
            List[EmotionNode]: 所有文本的情绪节点列表（扁平化）
        """
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]
        
        if len(texts) != len(text_ids):
            raise ValueError(f"Length mismatch: texts ({len(texts)}) != text_ids ({len(text_ids)})")
        
        all_nodes = []
        
        for text, text_id in zip(texts, text_ids):
            try:
                nodes = self.process(text, text_id)
                all_nodes.extend(nodes)
            except Exception as e:
                logger.error(f"Failed to process text_id '{text_id}': {e}")
                continue
        
        logger.info(
            f"Batch processing completed: {len(all_nodes)} nodes from {len(texts)} texts"
        )
        
        return all_nodes

