"""
Emotion V3 类 - 使用Emos模型

使用emos模型直接在一个句子中提取多个实体的情绪向量（256维hidden_state）。
不需要改写句子，不需要生成情感描述，直接使用原始句子+实体span。

完整流程：
1. 从文本中抽取实体
2. 使用emos模型直接提取实体在句子中的情绪向量（256维，可配置）
3. 返回 EmotionNode 列表

模型配置：
- 基座模型: Qwen/Qwen2.5-7B-Instruct (7B参数)
- 向量维度: 256维（可配置为64/128/256/512）
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from utils.entitiy import Entity
from hipporag.utils.logging_utils import get_logger
from particle.emotion_cache import EmotionCache
from particle.emos_wrapper import EmosWrapper

logger = get_logger(__name__)

# 复用EmotionNode结构
from particle.emotion_v2 import EmotionNode


class EmotionV3:
    """
    Emotion V3 类 - 使用Emos模型
    
    完整流程：
    1. 从文本中抽取实体
    2. 使用emos模型直接提取实体在句子中的情绪向量（256维hidden_state，可配置）
    3. 返回 EmotionNode 列表
    
    与EmotionV2的区别：
    - 不需要生成情感描述（Sentence类）
    - 不需要调用embedding API
    - 直接使用emos模型提取实体级别的情绪向量
    """
    
    def __init__(
        self,
        emos_checkpoint_path: Optional[str] = None,
        entity_extractor=None,
        enable_cache: bool = True,
        cache_dir: str = "./emotion_cache",
        device: str = "cpu",
        model_name: Optional[str] = None,  # 模型名称（可选，默认从checkpoint读取）
        use_8bit_quantization: bool = False  # 8-bit量化（用于8B模型）
    ):
        """
        初始化 EmotionV3 类

        Args:
            emos_checkpoint_path: emos模型checkpoint路径，如果为None则尝试默认路径
            entity_extractor: Entity 实例（可选），如果为 None 则自动创建
            enable_cache: 是否启用缓存
            cache_dir: 缓存目录路径
            device: 设备 (cpu/cuda)
            model_name: 模型名称（可选，默认从checkpoint自动读取）
            use_8bit_quantization: 是否使用8-bit量化（推荐用于8B模型以节省显存）
        """
        from llm.config import DEFAULT_MODEL

        # 初始化emos模型包装器
        try:
            # 如果没有提供model_name，尝试从checkpoint路径推断
            if model_name is None and emos_checkpoint_path:
                # 检查checkpoint路径是否包含模型信息
                checkpoint_str = str(emos_checkpoint_path)
                if "qwen" in checkpoint_str.lower() or "8b" in checkpoint_str.lower():
                    # 尝试使用Qwen模型（但这可能不对，因为best_model_stage2.pt是RoBERTa训练的）
                    pass
                # 默认使用roberta-base（因为best_model_stage2.pt是用RoBERTa训练的）
                if model_name is None:
                    model_name = "roberta-base"
            
            self.emos_wrapper = EmosWrapper(
                checkpoint_path=emos_checkpoint_path,
                device=device,
                model_name=model_name,
                use_8bit_quantization=use_8bit_quantization
            )
            logger.info("✅ EmosWrapper初始化成功")
        except Exception as e:
            logger.error(f"❌ EmosWrapper初始化失败: {e}")
            raise

        # 初始化实体抽取器
        if entity_extractor is None:
            self.entity_extractor = Entity(model_name=DEFAULT_MODEL)
        else:
            self.entity_extractor = entity_extractor

        # 初始化缓存管理器
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = EmotionCache(cache_dir=cache_dir)
            logger.info(f"EmotionV3 caching enabled: cache_dir={cache_dir}")
        else:
            self.cache = None
            logger.info("EmotionV3 caching disabled")

        logger.info(
            f"EmotionV3 initialized with device: {device}"
        )

    def process(
        self,
        text: str,
        text_id: str,
        entities: Optional[List[str]] = None,
        entity_positions: Optional[List[dict]] = None
    ) -> List[EmotionNode]:
        """
        处理文本，生成情绪节点列表
        
        完整流程：
        1. 抽取实体（如果未提供）
        2. 使用emos模型直接提取实体在句子中的情绪向量（64维）
        3. 返回 EmotionNode 列表
        
        Args:
            text: 原始文本
            text_id: 原文本 ID（用于映射关系）
            entities: 实体列表（可选），如果为 None 则自动抽取
            entity_positions: 实体位置信息列表（可选），每个元素包含{'char_start': int, 'char_end': int}
        
        Returns:
            List[EmotionNode]: 情绪节点列表
        """
        # Step 0: 检查空文本
        if not text or not text.strip():
            logger.warning("=" * 80)
            logger.warning(f"[EmotionV3.process] 输入文本为空，跳过处理")
            logger.warning(f"  text_id: {text_id}")
            logger.warning(f"  text_length: {len(text) if text else 0} 字符")
            logger.warning("=" * 80)
            return []

        # Step 1: 抽取实体（如果未提供）
        logger.info("=" * 80)
        logger.info(f"[EmotionV3.process] 开始处理文本")
        logger.info(f"  输入 - text_id: {text_id}")
        logger.info(f"  输入 - text: {text[:200]}{'...' if len(text) > 200 else ''}")
        logger.info(f"  输入 - text_length: {len(text)} 字符")
        logger.info(f"  输入 - entities: {entities if entities is not None else 'None (将自动抽取)'}")
        
        if entities is None:
            try:
                logger.info(f"[EmotionV3.process] 开始抽取实体...")
                entities = self.entity_extractor.extract_entities(text)
                logger.info(f"[EmotionV3.process] 实体抽取完成")
                logger.info(f"  抽取结果 - 实体数量: {len(entities)}")
                logger.info(f"  抽取结果 - 实体列表: {entities}")
                if not entities:
                    logger.warning(f"[EmotionV3.process] 警告: 未从文本中提取到任何实体")
                    logger.warning(f"  文本内容: {text[:200]}{'...' if len(text) > 200 else ''}")
                    return []
            except Exception as e:
                logger.error(f"[EmotionV3.process] 实体抽取失败: {e}")
                return []
        
        if not entities:
            logger.warning(f"[EmotionV3.process] 处理终止: 没有实体可处理")
            return []
        
        # Step 2: 使用emos模型直接提取实体的情绪向量（64维）
        logger.info(f"[EmotionV3.process] 开始提取实体情绪向量（使用emos模型）...")
        logger.info(f"  实体数量: {len(entities)}")
        logger.info(f"  实体列表: {entities}")
        
        try:
            # 使用emos模型批量提取实体情绪向量
            emotion_vectors_dict = self.emos_wrapper.extract_entities_emotion_vectors(
                text=text,
                entities=entities,
                entity_positions=entity_positions
            )
            
            logger.info(f"[EmotionV3.process] 情绪向量提取完成")
            logger.info(f"  成功提取: {len([v for v in emotion_vectors_dict.values() if v.sum() != 0])}/{len(entities)} 个向量")
            
        except Exception as e:
            logger.error(f"[EmotionV3.process] 情绪向量提取失败: {e}")
            import traceback
            logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
            return []
        
        # Step 3: 创建 EmotionNode 列表
        logger.info(f"[EmotionV3.process] 开始创建EmotionNode列表...")
        
        nodes = []
        for idx, entity in enumerate(entities):
            emotion_vector = emotion_vectors_dict.get(entity)
            
            if emotion_vector is None or emotion_vector.size == 0:
                logger.warning(f"[EmotionV3.process] 跳过实体 '{entity}': 情绪向量为空")
                continue
            
            # 验证向量维度（支持64, 128, 256, 512等）
            expected_dims = [64, 128, 256, 512]
            actual_dim = len(emotion_vector)
            if actual_dim not in expected_dims:
                logger.warning(f"[EmotionV3.process] 实体 '{entity}' 的情绪向量维度异常: {actual_dim}，预期为 {expected_dims}")
                # 不跳过，继续处理（可能是模型输出维度不同）
                logger.info(f"[EmotionV3.process] 继续处理，使用实际维度: {actual_dim}")
            
            try:
                logger.debug(f"[EmotionV3.process] 处理实体 {idx+1}/{len(entities)}: '{entity}'")
                
                # 生成实体 ID（兼容 HippoRAG 格式）
                from hipporag.utils.misc_utils import compute_mdhash_id
                standard_entity_id = compute_mdhash_id(content=entity.lower(), prefix="entity-")
                
                # 生成粒子唯一 ID
                particle_entity_id = f"{text_id}_{standard_entity_id}"
                
                # 计算情绪强度（使用向量的L2范数作为强度）
                # 注意：emos模型返回的是normalized向量（mu），范数通常在0-1之间
                intensity = float(np.linalg.norm(emotion_vector))
                
                # 创建 EmotionNode（存储原始向量，不归一化）
                node = EmotionNode(
                    entity_id=standard_entity_id,
                    entity=entity,
                    emotion_vector=emotion_vector,  # 64维向量
                    text_id=text_id,
                    intensity=intensity,
                    raw_description=""  # 不使用情感描述
                )
                
                nodes.append(node)
                
                logger.info(
                    f"[EmotionV3.process] 成功创建 EmotionNode: "
                    f"entity_id={standard_entity_id}, entity={entity}, "
                    f"vector_shape={emotion_vector.shape}, "
                    f"vector_norm={np.linalg.norm(emotion_vector):.6f}, "
                    f"intensity={intensity:.4f}"
                )
                
            except Exception as e:
                logger.error(f"[EmotionV3.process] 处理实体 '{entity}' 失败: {e}")
                import traceback
                logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
                continue
        
        logger.info(f"[EmotionV3.process] 处理完成")
        logger.info(f"  成功处理: {len(nodes)}/{len(entities)} 个实体")
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
