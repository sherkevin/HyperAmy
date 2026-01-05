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
from particle.emotion_cache import EmotionCache

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
        sentence_processor=None,
        enable_cache: bool = True,
        cache_dir: str = "./emotion_cache",
        use_batch_prompt: bool = True
    ):
        """
        初始化 EmotionV2 类

        Args:
            model_name: LLM 模型名称，用于 Sentence 类
            embedding_model_name: 嵌入模型名称，用于生成情绪嵌入向量
            entity_extractor: Entity 实例（可选），如果为 None 则自动创建
            sentence_processor: Sentence 实例（可选），如果为 None 则自动创建
            enable_cache: 是否启用缓存（优化方案三）
            cache_dir: 缓存目录路径
            use_batch_prompt: 是否使用批量 Prompt（优化方案六A）
        """
        from llm.config import DEFAULT_MODEL

        self.model_name = model_name or DEFAULT_MODEL
        self.embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL

        # 初始化组件
        if entity_extractor is None:
            self.entity_extractor = Entity(model_name=self.model_name)
        else:
            self.entity_extractor = entity_extractor

        # 初始化缓存管理器（优化方案三）
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = EmotionCache(cache_dir=cache_dir)
            logger.info(f"EmotionV2 caching enabled: cache_dir={cache_dir}")
        else:
            self.cache = None
            logger.info("EmotionV2 caching disabled")

        # 批量 Prompt 配置（优化方案六A）
        self.use_batch_prompt = use_batch_prompt

        # 创建 Sentence 处理器（如果需要，传入缓存实例）
        if sentence_processor is None:
            self.sentence_processor = Sentence(model_name=self.model_name, cache=self.cache)
        else:
            self.sentence_processor = sentence_processor
            # 如果提供了自定义 sentence_processor，也更新其缓存引用
            if self.enable_cache and hasattr(sentence_processor, 'cache'):
                sentence_processor.cache = self.cache

        logger.info(
            f"EmotionV2 initialized with model: {self.model_name}, "
            f"embedding_model: {self.embedding_model_name}, "
            f"batch_prompt: {self.use_batch_prompt}"
        )
    
    def _get_emotion_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的情绪嵌入向量

        使用 embedding API 将文本转换为高维稠密向量
        支持缓存（优化方案三）

        Args:
            text: 输入文本（情感描述句子）

        Returns:
            numpy.ndarray: 情绪嵌入向量
        """
        # 检查缓存（优化方案三）
        if self.enable_cache and self.cache:
            cached_embedding = self.cache.get_cached_embedding(text)
            if cached_embedding is not None:
                logger.debug(f"Cache HIT for embedding: {text[:50]}...")
                return cached_embedding

        # 缓存未命中，调用API生成
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

            # 保存到缓存（优化方案三）
            if self.enable_cache and self.cache:
                self.cache.save_embedding(text, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Failed to get emotion embedding for text '{text[:50]}...': {e}")
            raise

    def _batch_get_emotion_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量获取多个文本的情绪嵌入向量

        优化方案二：一次API调用处理所有文本，显著降低总耗时
        优化方案三：支持缓存检查

        Args:
            texts: 输入文本列表（情感描述句子列表）

        Returns:
            List[np.ndarray]: 情绪嵌入向量列表
        """
        if not texts:
            return []

        # 初始化结果列表
        embeddings = [None] * len(texts)

        # 第一步：检查缓存（优化方案三）
        cache_hits = []
        cache_miss_indices = []

        if self.enable_cache and self.cache:
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    embeddings[i] = np.array([])
                    continue

                cached_embedding = self.cache.get_cached_embedding(text)
                if cached_embedding is not None:
                    embeddings[i] = cached_embedding
                    cache_hits.append(i)
                else:
                    cache_miss_indices.append(i)

            logger.info(f"Batch embedding: {len(cache_hits)} cache hits, {len(cache_miss_indices)} cache misses")
        else:
            # 缓存未启用，所有文本都需要生成
            cache_miss_indices = [i for i, text in enumerate(texts) if text and text.strip()]

        # 第二步：批量生成未缓存的嵌入
        if cache_miss_indices:
            miss_texts = [texts[i] for i in cache_miss_indices]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }

            # 批量调用：一次处理所有未缓存的文本
            payload = {
                "model": self.embedding_model_name,
                "input": miss_texts,
                "encoding_format": "float"
            }

            try:
                logger.debug(f"Batch embedding API request for {len(miss_texts)} texts...")
                response = requests.post(API_URL_EMBEDDINGS, headers=headers, json=payload)
                response.raise_for_status()

                result = response.json()

                # 提取所有 embedding 向量
                if isinstance(result.get("data"), list) and len(result["data"]) > 0:
                    for item in result["data"]:
                        idx = item.get("index", len(miss_texts))
                        if idx < len(cache_miss_indices):
                            original_idx = cache_miss_indices[idx]
                            embedding = np.array(item["embedding"])
                            embeddings[original_idx] = embedding

                            # 保存到缓存（优化方案三）
                            if self.enable_cache and self.cache:
                                self.cache.save_embedding(miss_texts[idx], embedding)

                    logger.info(f"Generated {len([e for e in embeddings if e is not None and e.size > 0])} embeddings (batch + cache)")
                else:
                    raise ValueError(f"Unexpected API response format: {result}")

            except Exception as e:
                logger.error(f"Failed to get batch emotion embeddings: {e}")
                # 失败时回退到单个调用
                logger.warning("Falling back to individual embedding calls...")
                for idx in cache_miss_indices:
                    text = texts[idx]
                    try:
                        embedding = self._get_emotion_embedding(text)
                        embeddings[idx] = embedding
                    except Exception as e2:
                        logger.error(f"Failed to get embedding for text at index {idx}: {e2}")
                        embeddings[idx] = np.array([])

        # 填充None为空向量
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = np.array([])
                logger.warning(f"Failed to get embedding for text at index {i}")

        return embeddings

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
        
        # Step 2: 为每个实体生成情感描述（优化：批量Prompt或并行）
        if self.use_batch_prompt:
            logger.info(f"[EmotionV2.process] 开始为实体生成情感描述（批量Prompt模式）...")
        else:
            logger.info(f"[EmotionV2.process] 开始为实体生成情感描述（并行模式）...")
        logger.info(f"  实体数量: {len(entities)}")
        logger.info(f"  实体列表: {entities}")
        try:
            # 根据配置选择批量或并行模式（优化方案六A vs 方案一）
            if self.use_batch_prompt:
                affective_descriptions = self.sentence_processor.generate_affective_descriptions_batch(
                    sentence=text,
                    entities=entities
                )
            else:
                affective_descriptions = self.sentence_processor.generate_affective_descriptions_parallel(
                    sentence=text,
                    entities=entities,
                    max_workers=5  # 并行度：建议3-5
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
        
        # Step 3: 批量生成情绪嵌入向量，生成 EmotionNode 列表（优化方案二）
        logger.info(f"[EmotionV2.process] 开始批量生成情绪嵌入向量...")
        logger.info(f"  使用批量Embedding调用（优化方案二）")

        # 准备所有描述文本
        descriptions_list = [affective_descriptions.get(entity, "") for entity in entities]

        # 批量生成嵌入向量
        try:
            emotion_vectors = self._batch_get_emotion_embeddings(descriptions_list)
            logger.info(f"[EmotionV2.process] 批量嵌入生成完成，共 {len(emotion_vectors)} 个向量")
        except Exception as e:
            logger.error(f"[EmotionV2.process] 批量嵌入生成失败")
            logger.error(f"  错误信息: {str(e)}")
            import traceback
            logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
            return []

        # 创建 EmotionNode 列表
        nodes = []
        for idx, entity in enumerate(entities):
            description = affective_descriptions.get(entity, "")
            emotion_vector = emotion_vectors[idx]

            if not description:
                logger.warning(f"[EmotionV2.process] 跳过实体 '{entity}': 情感描述为空")
                logger.warning(f"  实体索引: {idx}/{len(entities)}")
                logger.warning(f"  文本内容: {text[:200]}{'...' if len(text) > 200 else ''}")
                continue

            if emotion_vector.size == 0:
                logger.warning(f"[EmotionV2.process] 跳过实体 '{entity}': 嵌入向量为空")
                logger.warning(f"  实体索引: {idx}/{len(entities)}")
                logger.warning(f"  情感描述: {description[:150]}{'...' if len(description) > 150 else ''}")
                continue

            try:
                logger.debug(f"[EmotionV2.process] 处理实体 {idx+1}/{len(entities)}: '{entity}'")
                logger.debug(f"  情感描述: {description[:150]}{'...' if len(description) > 150 else ''}")

                # 生成实体 ID（兼容 HippoRAG 格式：MD5 hash with "entity-" prefix）
                # 这样可以确保 Amygdala 和 HippoRAG 使用相同的实体 ID
                # 重要：统一转换为小写，因为 HippoRAG 的 OpenIE 会将实体小写化
                from hipporag.utils.misc_utils import compute_mdhash_id
                standard_entity_id = compute_mdhash_id(content=entity.lower(), prefix="entity-")

                # 生成粒子唯一 ID（包含 text_id，避免不同文档中的同名实体冲突）
                particle_entity_id = f"{text_id}_{standard_entity_id}"

                # 创建 EmotionNode
                # 注意：entity_id 使用标准格式（用于与 HippoRAG 匹配）
                # 但实际上每个粒子有唯一的 particle_entity_id（在 ParticleEntity 中使用）
                node = EmotionNode(
                    entity_id=standard_entity_id,  # 使用标准 entity_id
                    entity=entity,
                    emotion_vector=emotion_vector,
                    text_id=text_id
                )

                nodes.append(node)

                logger.info(
                    f"[EmotionV2.process] 成功创建 EmotionNode: "
                    f"entity_id={standard_entity_id}, entity={entity}, "
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

