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
from .purity import Purity
from .speed import Speed
from .temperature import Temperature
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ParticleEntity:
    """
    粒子实体结构体（基于自由能原理 + Soft Label 强度）

    包含实体的所有属性：
    - entity_id: 实体 ID（唯一标识）
    - entity: 实体名称
    - text_id: 原文本 ID（映射关系）
    - emotion_vector: 归一化方向向量 μ
    - weight: 粒子质量/模长 κ = 1.0 + α × I_raw
    - speed: 初始速度 v_0（等于模长）
    - temperature: 初始温度 T_0
    - purity: 归一化纯度 [0, 1]
    - tau_v: 速度衰减时间常数（秒）
    - tau_T: 温度冷却时间常数（秒）
    - born: 生成时间戳
    - intensity: 原始情绪强度 I_raw ∈ [0, 1]（Soft Label 最大值）
    """
    entity_id: str
    entity: str
    text_id: str
    emotion_vector: np.ndarray  # 归一化方向 μ
    weight: float  # 模长 κ（作为质量）
    speed: float  # 初始速度 v_0 = κ
    temperature: float  # 初始温度 T_0
    purity: float  # 归一化纯度 [0, 1]
    tau_v: float  # 速度衰减时间常数
    tau_T: float  # 温度冷却时间常数
    born: float
    intensity: float = 0.0  # 情绪强度 I_raw


class Particle:
    """
    粒子聚合类（基于自由能原理）

    整合 emotion、purity、speed、temperature 模块，提供统一的处理接口。
    """

    def __init__(
        self,
        model_name=None,
        embedding_model_name=None,
        entity_extractor=None,
        sentence_processor=None,
        # 热力学参数
        T_min: float = 0.1,
        T_max: float = 1.0,
        alpha: float = 0.5,
        tau_base: float = 86400.0,
        beta: float = 1.0,
        gamma: float = 2.0,
        # 模长参数（基于 Soft Label 强度）
        intensity_alpha: float = 50.0
    ):
        """
        初始化 Particle 类

        Args:
            model_name: LLM 模型名称
            embedding_model_name: 嵌入模型名称
            entity_extractor: Entity 实例（可选）
            sentence_processor: Sentence 实例（可选）
            T_min: 最小温度（有序态）
            T_max: 最大温度（无序态）
            alpha: 纯度对速度的影响系数
            tau_base: 基准时间常数（秒，默认1天）
            beta: 温度冷却系数
            gamma: 速度衰减系数
            intensity_alpha: 模长系数 κ = 1.0 + α × I_raw（默认 50.0）
        """
        # 初始化各个模块
        self.emotion_v2 = EmotionV2(
            model_name=model_name,
            embedding_model_name=embedding_model_name,
            entity_extractor=entity_extractor,
            sentence_processor=sentence_processor
        )
        self.purity = Purity()
        self.speed = Speed(alpha=alpha)
        self.temperature = Temperature(T_min=T_min, T_max=T_max)

        # 存储热力学参数
        self.T_min = T_min
        self.T_max = T_max
        self.alpha = alpha
        self.tau_base = tau_base
        self.beta = beta
        self.gamma = gamma
        self.intensity_alpha = intensity_alpha

        # 初始化情绪向量处理器
        from particle.emotion_vector_processor import EmotionVectorProcessor
        self.vector_processor = EmotionVectorProcessor(
            alpha=intensity_alpha,
            min_modulus=1.0
        )

        logger.info(
            f"Particle module initialized (free energy parameters: "
            f"T_min={T_min}, T_max={T_max}, alpha={alpha}, "
            f"tau_base={tau_base}, beta={beta}, gamma={gamma}, "
            f"intensity_alpha={intensity_alpha})"
        )
    
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
        entity_ids = [node.entity_id for node in emotion_nodes]
        entity_names = [node.entity for node in emotion_nodes]
        emotion_vectors = [node.emotion_vector for node in emotion_nodes]
        intensities = [node.intensity for node in emotion_nodes]

        # Step 3: 使用 EmotionVectorProcessor 计算方向和模长（解耦设计）
        processed_vectors = self.vector_processor.process_batch(
            raw_vectors=emotion_vectors,
            intensities=intensities,
            normalize=True
        )

        # 提取方向和模长
        directions = [pv.direction for pv in processed_vectors]
        moduli = [pv.modulus for pv in processed_vectors]
        states = [pv.state for pv in processed_vectors]

        logger.info(
            f"Processed emotion vectors: moduli={moduli}, states={states}"
        )

        # Step 4: 计算纯度（基于原始向量，用于热力学）
        purities = []
        for vec in emotion_vectors:
            purity_norm = self.purity.compute_normalized(vec)
            purities.append(purity_norm)
        logger.info(f"Computed purities: {purities}")

        # Step 5: 计算温度（速度现在由模长直接决定）
        try:
            temperatures = self.temperature.compute(
                entity_ids=entity_ids,
                emotion_vectors=emotion_vectors,
                text_id=text_id
            )
        except Exception as e:
            logger.error(f"Failed to compute temperatures: {e}")
            temperatures = [0.5] * len(entity_ids)

        # Step 6: 聚合所有信息，生成 ParticleEntity 列表
        particles = []
        for i, (standard_entity_id, entity_name, direction, modulus, temp, purity, state) in enumerate(
            zip(entity_ids, entity_names, directions, moduli, temperatures, purities, states)
        ):
            # 生成唯一的粒子 ID
            unique_particle_id = f"{text_id}_{standard_entity_id}"

            # 计算时间常数
            tau_v = self.tau_base * (1.0 + self.gamma * purity)
            tau_T = self.tau_base * (1.0 + self.beta * purity)

            particle = ParticleEntity(
                entity_id=unique_particle_id,
                entity=entity_name,
                text_id=text_id,
                emotion_vector=direction,  # 归一化方向 μ
                weight=modulus,  # 模长 κ = 1.0 + α × I_raw（作为质量）
                speed=modulus,  # 初始速度 = 模长
                temperature=temp,  # 初始温度 T_0
                purity=purity,  # 归一化纯度
                tau_v=tau_v,  # 速度衰减时间常数
                tau_T=tau_T,  # 温度冷却时间常数
                born=t_born,
                intensity=intensities[i]  # 情绪强度 I_raw
            )
            particles.append(particle)

            logger.debug(
                f"Created particle: entity_id={unique_particle_id}, entity={entity_name}, "
                f"modulus={modulus:.2f} ({state}), intensity={intensities[i]:.3f}, "
                f"temperature={temp:.4f}, purity={purity:.4f}, tau_v={tau_v:.2f}, tau_T={tau_T:.2f}"
            )

        logger.info(
            f"Successfully created {len(particles)} particles with free energy properties "
            f"for text_id: {text_id}"
        )

        return particles
