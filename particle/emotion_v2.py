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
        # Step 0: 检查空文本
        if not text or not text.strip():
            logger.warning("=" * 80)
            logger.warning(f"[EmotionV2.process] 输入文本为空，跳过处理")
            logger.warning(f"  text_id: {text_id}")
            logger.warning(f"  text_length: {len(text) if text else 0} 字符")
            logger.warning("=" * 80)
            return []

        # Step 1: 抽取实体（如果未提供）
        logger.info("=" * 80)
        logger.info(f"[EmotionV2.process] 开始处理文本")
        logger.info(f"  输入 - text_id: {text_id}")
        logger.info(f"  输入 - text: {text[:200]}{'...' if len(text) > 200 else ''}")
        logger.info(f"  输入 - text_length: {len(text)} 字符")
        logger.info(f"  输入 - entities: {entities if entities is not None else 'None (将自动抽取)'}")
        
        if entities is None:
            try:
                logger.info(f"[EmotionV2.process] 开始抽取实体...")
                entities = self.entity_extractor.extract_entities(text)
                logger.info(f"[EmotionV2.process] 实体抽取完成")
                logger.info(f"  抽取结果 - 实体数量: {len(entities)}")
                logger.info(f"  抽取结果 - 实体列表: {entities}")
                if not entities:
                    logger.warning(f"[EmotionV2.process] 警告: 未从文本中提取到任何实体")
                    logger.warning(f"  文本内容: {text[:200]}{'...' if len(text) > 200 else ''}")
                    logger.warning(f"  可能原因: 1) 文本中确实没有命名实体 2) 实体提取器无法识别该类型的实体")
            except Exception as e:
                logger.error(f"[EmotionV2.process] 实体抽取失败")
                logger.error(f"  文本内容: {text[:200]}{'...' if len(text) > 200 else ''}")
                logger.error(f"  text_id: {text_id}")
                logger.error(f"  错误信息: {str(e)}")
                import traceback
                logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
                return []  # 如果抽取失败，返回空列表
        
        if not entities:
            logger.warning(f"[EmotionV2.process] 处理终止: 没有实体可处理")
            logger.warning(f"  text_id: {text_id}")
            logger.warning(f"  文本内容: {text[:200]}{'...' if len(text) > 200 else ''}")
            logger.warning("=" * 80)
            return []
        
        # Step 2: 为每个实体生成情感描述
        logger.info(f"[EmotionV2.process] 开始为实体生成情感描述...")
        logger.info(f"  实体数量: {len(entities)}")
        logger.info(f"  实体列表: {entities}")
        try:
            affective_descriptions = self.sentence_processor.generate_affective_descriptions(
                sentence=text,
                entities=entities
            )
            successful_descriptions = [d for d in affective_descriptions.values() if d]
            logger.info(f"[EmotionV2.process] 情感描述生成完成")
            logger.info(f"  成功生成: {len(successful_descriptions)}/{len(entities)} 个描述")
            for entity, desc in affective_descriptions.items():
                if desc:
                    logger.debug(f"    实体 '{entity}': {desc[:100]}{'...' if len(desc) > 100 else ''}")
                else:
                    logger.warning(f"    实体 '{entity}': 描述为空")
        except Exception as e:
            logger.error(f"[EmotionV2.process] 情感描述生成失败")
            logger.error(f"  文本内容: {text[:200]}{'...' if len(text) > 200 else ''}")
            logger.error(f"  实体列表: {entities}")
            logger.error(f"  错误信息: {str(e)}")
            import traceback
            logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
            return []
        
        # Step 3: 对每个情感描述进行嵌入，生成 EmotionNode 列表
        logger.info(f"[EmotionV2.process] 开始生成情绪嵌入向量...")
        nodes = []
        for idx, entity in enumerate(entities):
            description = affective_descriptions.get(entity, "")
            
            if not description:
                logger.warning(f"[EmotionV2.process] 跳过实体 '{entity}': 情感描述为空")
                logger.warning(f"  实体索引: {idx}/{len(entities)}")
                logger.warning(f"  文本内容: {text[:200]}{'...' if len(text) > 200 else ''}")
                continue
            
            try:
                logger.debug(f"[EmotionV2.process] 处理实体 {idx+1}/{len(entities)}: '{entity}'")
                logger.debug(f"  情感描述: {description[:150]}{'...' if len(description) > 150 else ''}")
                
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
                
                logger.info(
                    f"[EmotionV2.process] 成功创建 EmotionNode: "
                    f"entity_id={entity_id}, entity={entity}, "
                    f"vector_shape={emotion_vector.shape}, "
                    f"vector_norm={np.linalg.norm(emotion_vector):.6f}"
                )
                
            except Exception as e:
                logger.error(f"[EmotionV2.process] 处理实体 '{entity}' 失败")
                logger.error(f"  实体索引: {idx}/{len(entities)}")
                logger.error(f"  情感描述: {description[:150]}{'...' if len(description) > 150 else ''}")
                logger.error(f"  错误信息: {str(e)}")
                import traceback
                logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
                continue  # 跳过失败的实体，继续处理其他实体
        
        logger.info(f"[EmotionV2.process] 处理完成")
        logger.info(f"  成功处理: {len(nodes)}/{len(entities)} 个实体")
        logger.info(f"  生成的 EmotionNode 列表:")
        for i, node in enumerate(nodes, 1):
            logger.info(
                f"    {i}. entity_id={node.entity_id}, "
                f"entity={node.entity}, "
                f"vector_shape={node.emotion_vector.shape}"
            )
        logger.info("=" * 80)
        
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

