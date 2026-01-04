"""
Particle 类

聚合模块：整合 emotion、speed、temperature 的逻辑。
输入：一段文本和对应的id
输出：一个列表，每个值为一个结构体，包含实体的id、到文本id的映射关系、情绪向量、速度、温度等属性。
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import time

from .emotion_v2 import EmotionV2, EmotionNode
from .speed import Speed
from .temperature import Temperature
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ParticleEntity:
    """
    粒子实体结构体
    
    包含实体的所有属性：
    - entity_id: 实体 ID（唯一标识）
    - entity: 实体名称
    - text_id: 原文本 ID（映射关系）
    - emotion_vector: 情绪向量（归一化后的方向向量）
    - weight: 粒子质量（初始情绪向量的模长）
    - speed: 速度/强度
    - temperature: 温度/熵
    - born: 生成时间戳
    """
    entity_id: str
    entity: str
    text_id: str
    emotion_vector: np.ndarray  # 归一化后的方向向量
    weight: float  # 粒子质量（初始情绪向量的模长）
    speed: float
    temperature: float
    born: float


class Particle:
    """
    粒子聚合类
    
    整合 emotion、speed、temperature 模块，提供统一的处理接口。
    """
    
    def __init__(
        self,
        model_name=None,
        embedding_model_name=None,
        entity_extractor=None,
        sentence_processor=None
    ):
        """
        初始化 Particle 类
        
        Args:
            model_name: LLM 模型名称
            embedding_model_name: 嵌入模型名称
            entity_extractor: Entity 实例（可选）
            sentence_processor: Sentence 实例（可选）
        """
        # 初始化各个模块
        self.emotion_v2 = EmotionV2(
            model_name=model_name,
            embedding_model_name=embedding_model_name,
            entity_extractor=entity_extractor,
            sentence_processor=sentence_processor
        )
        self.speed = Speed()
        self.temperature = Temperature()
        
        logger.info("Particle module initialized")
    
    def process(
        self,
        text: str,
        text_id: str,
        entities: Optional[List[str]] = None
    ) -> List[ParticleEntity]:
        """
        处理文本，生成粒子实体列表
        
        Args:
            text: 输入文本
            text_id: 文本 ID
            entities: 实体列表（可选），如果为 None 则自动抽取
        
        Returns:
            List[ParticleEntity]: 粒子实体列表
        """
        t_born = time.time()
        
        logger.info("=" * 80)
        logger.info(f"[Particle.process] 开始处理文本")
        logger.info(f"  输入 - text_id: {text_id}")
        logger.info(f"  输入 - text: {text[:200]}{'...' if len(text) > 200 else ''}")
        logger.info(f"  输入 - text_length: {len(text)} 字符")
        logger.info(f"  输入 - entities: {entities if entities is not None else 'None (将自动抽取)'}")
        
        # Step 1: 提取情绪节点
        try:
            emotion_nodes = self.emotion_v2.process(
                text=text,
                text_id=text_id,
                entities=entities
            )
            logger.info(f"[Particle.process] 情绪节点提取完成")
            logger.info(f"  提取结果 - emotion_nodes数量: {len(emotion_nodes)}")
            if emotion_nodes:
                logger.info(f"  提取结果 - emotion_nodes详情:")
                for i, node in enumerate(emotion_nodes, 1):
                    logger.info(
                        f"    {i}. entity_id={node.entity_id}, "
                        f"entity={node.entity}, "
                        f"vector_shape={node.emotion_vector.shape}"
                    )
        except Exception as e:
            logger.error(f"[Particle.process] 情绪节点提取失败")
            logger.error(f"  文本内容: {text[:200]}{'...' if len(text) > 200 else ''}")
            logger.error(f"  text_id: {text_id}")
            logger.error(f"  错误信息: {str(e)}")
            import traceback
            logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
            return []
        
        if not emotion_nodes:
            logger.warning(f"[Particle.process] 处理终止: 没有情绪节点")
            logger.warning(f"  text_id: {text_id}")
            logger.warning(f"  文本内容: {text[:200]}{'...' if len(text) > 200 else ''}")
            logger.warning(f"  可能原因: 1) 未提取到实体 2) 情感描述生成失败 3) 情绪嵌入失败")
            logger.warning("=" * 80)
            return []
        
        # Step 2: 提取实体信息
        entity_ids = [node.entity_id for node in emotion_nodes]  # 标准 entity_id（用于与 HippoRAG 匹配）
        entity_names = [node.entity for node in emotion_nodes]
        emotion_vectors = [node.emotion_vector for node in emotion_nodes]

        # Step 3: 计算速度和温度
        try:
            speeds = self.speed.compute(
                entity_ids=entity_ids,
                emotion_vectors=emotion_vectors,
                text_id=text_id
            )
        except Exception as e:
            logger.error(f"Failed to compute speeds: {e}")
            # 使用默认值
            speeds = [0.5] * len(entity_ids)

        try:
            temperatures = self.temperature.compute(
                entity_ids=entity_ids,
                emotion_vectors=emotion_vectors,
                text_id=text_id
            )
        except Exception as e:
            logger.error(f"Failed to compute temperatures: {e}")
            # 使用默认值
            temperatures = [0.5] * len(entity_ids)

        # Step 4: 聚合所有信息，生成 ParticleEntity 列表
        particles = []
        for i, (standard_entity_id, entity_name, emotion_vector, speed, temp) in enumerate(
            zip(entity_ids, entity_names, emotion_vectors, speeds, temperatures)
        ):
            # 计算权重（初始情绪向量的模长）
            weight = float(np.linalg.norm(emotion_vector))

            # 归一化情绪向量（存储方向）
            if weight > 1e-9:
                normalized_vector = emotion_vector / weight
            else:
                normalized_vector = emotion_vector.copy()
                weight = 0.0

            # 生成唯一的粒子 ID（避免不同文档中的同名实体冲突）
            # 使用格式：text_id_standard_entity_id
            unique_particle_id = f"{text_id}_{standard_entity_id}"

            particle = ParticleEntity(
                entity_id=unique_particle_id,  # 粒子唯一 ID
                entity=entity_name,
                text_id=text_id,
                emotion_vector=normalized_vector,  # 归一化后的方向向量
                weight=weight,  # 存储原始模长作为质量
                speed=speed,
                temperature=temp,
                born=t_born
            )
            particles.append(particle)
            
            logger.debug(
                f"Created particle: entity_id={unique_particle_id}, entity={entity_name}, "
                f"speed={speed:.4f}, temperature={temp:.4f}, text_id={text_id}"
            )
        
        logger.info(
            f"Successfully created {len(particles)} particles for text_id: {text_id}"
        )
        
        return particles
