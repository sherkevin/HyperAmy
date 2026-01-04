"""
Sentence 类

实现 Step 2: 为每个实体生成情感视角描述 (Affective Perspective Description)
"""
from typing import List, Dict, Optional
from llm import create_client
from llm.config import API_URL_CHAT, DEFAULT_MODEL
from hipporag.utils.logging_utils import get_logger
from prompts.prompt_template_manager import PromptTemplateManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = get_logger(__name__)


class Sentence:
    """
    句子处理类
    
    功能：
    - Step 1: 实体识别（使用 Entity 类）
    - Step 2: 为每个实体生成情感视角描述
    """
    
    def __init__(self, model_name=None, cache=None):
        """
        初始化 Sentence 类

        Args:
            model_name: 使用的模型名称，如果为 None 则使用默认模型
            cache: 可选的 EmotionCache 实例，用于缓存情感描述
        """
        self.model_name = model_name or DEFAULT_MODEL

        # 创建 LLM 客户端（使用 normal 模式，Chat API）
        self.client = create_client(
            model_name=self.model_name,
            chat_api_url=API_URL_CHAT,
            mode="normal"
        )

        # 初始化 prompt 模板管理器
        self.prompt_template_manager = PromptTemplateManager()

        # 初始化缓存（可选）
        self.cache = cache

        logger.info(f"Sentence initialized with model: {self.model_name}, cache={'enabled' if cache else 'disabled'}")
    
    def generate_affective_description(self, sentence: str, entity: str) -> str:
        """
        为单个实体生成情感视角描述

        Prompt 逻辑：基于句子 S，生成关于 e_i 的纯情绪化描述。
        忽略事实细节，只保留情感色彩。

        支持缓存（优化方案三）

        Args:
            sentence: 原始句子 S
            entity: 实体 e_i

        Returns:
            str: 情感视角描述 d_i
        """
        # 检查缓存（优化方案三）
        if self.cache:
            cached_description = self.cache.get_cached_description(sentence, entity)
            if cached_description is not None:
                logger.debug(f"[Sentence] Cache HIT for description: entity='{entity}'")
                return cached_description

        # 使用模板管理器渲染 prompt
        prompt = self.prompt_template_manager.render(
            name='affective_description',
            sentence=sentence,
            entity=entity
        )

        try:
            logger.info(f"[Sentence.generate_affective_description] 开始为实体 '{entity}' 生成情感描述")
            logger.debug(f"  Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

            result = self.client.complete(
                query=prompt,
                max_tokens=50,  # 情绪词列表较短，减少 token 数
                temperature=0.7  # 稍高温度以增加描述的多样性
            )

            description = result.get_answer_text().strip()

            logger.info(
                f"[Sentence.generate_affective_description] 成功生成描述: {description[:100]}{'...' if len(description) > 100 else ''}"
            )

            # 保存到缓存（优化方案三）
            if self.cache:
                self.cache.save_description(sentence, entity, description)

            return description

        except Exception as e:
            logger.error(f"[Sentence.generate_affective_description] 生成失败")
            logger.error(f"  实体: {entity}")
            logger.error(f"  句子: {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
            logger.error(f"  错误信息: {str(e)}")
            import traceback
            logger.error(f"  错误堆栈:\n{traceback.format_exc()}")
            raise
    
    def generate_affective_descriptions(
        self, 
        sentence: str, 
        entities: List[str]
    ) -> Dict[str, str]:
        """
        为实体列表中的每个实体生成情感视角描述
        
        数学表达：d_i = LLM_rewrite(S, e_i)
        
        Args:
            sentence: 原始句子 S
            entities: 实体列表 E = {{e_1, e_2, ...}}
        
        Returns:
            dict: 实体到情感描述的映射 {{e_1: d_1, e_2: d_2, ...}}
        """
        descriptions = {}
        
        for entity in entities:
            try:
                description = self.generate_affective_description(sentence, entity)
                descriptions[entity] = description
            except Exception as e:
                logger.warning(f"Failed to generate description for entity '{entity}': {e}")
                descriptions[entity] = ""  # 失败时返回空字符串
        
        logger.info(
            f"Generated {len([d for d in descriptions.values() if d])} affective descriptions "
            f"for {len(entities)} entities"
        )
        
        return descriptions
    
    def process_sentence(
        self, 
        sentence: str, 
        entities: List[str] = None,
        entity_extractor=None
    ) -> Dict[str, any]:
        """
        完整处理句子：提取实体并生成情感描述
        
        Args:
            sentence: 原始句子 S
            entities: 实体列表（可选），如果为 None 则使用 entity_extractor 提取
            entity_extractor: Entity 实例（可选），用于提取实体
        
        Returns:
            dict: 包含 'sentence', 'entities', 'affective_descriptions' 的字典
        """
        # Step 1: 提取实体（如果未提供）
        if entities is None:
            if entity_extractor is None:
                from utils.entitiy import Entity
                entity_extractor = Entity()
            entities = entity_extractor.extract_entities(sentence)
        
        # Step 2: 生成情感描述
        affective_descriptions = self.generate_affective_descriptions(sentence, entities)
        
        return {
            "sentence": sentence,
            "entities": entities,
            "affective_descriptions": affective_descriptions
        }

    def generate_affective_descriptions_parallel(
        self,
        sentence: str,
        entities: List[str],
        max_workers: int = 5
    ) -> Dict[str, str]:
        """
        并行化版本：同时调用多个LLM生成情感描述

        优化方案一：使用线程池并行处理多个实体，显著降低总耗时

        Args:
            sentence: 原始句子 S
            entities: 实体列表 E = {e_1, e_2, ...}
            max_workers: 最大并行线程数（建议3-5，避免API限流）

        Returns:
            dict: 实体到情感描述的映射 {e_1: d_1, e_2: d_2, ...}
        """
        if not entities:
            return {}

        descriptions = {entity: "" for entity in entities}
        start_time = time.time()

        logger.info(f"[Sentence.generate_affective_descriptions_parallel] 开始并行生成情感描述")
        logger.info(f"  实体数量: {len(entities)}")
        logger.info(f"  并行度: {max_workers}")

        def generate_for_entity(entity):
            """为单个实体生成描述的包装函数"""
            try:
                description = self.generate_affective_description(sentence, entity)
                return entity, description, None
            except Exception as e:
                logger.warning(f"[generate_affective_descriptions_parallel] 实体 '{entity}' 生成失败: {e}")
                return entity, "", e

        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(generate_for_entity, entity): entity
                for entity in entities
            }

            # 收集结果
            completed = 0
            for future in as_completed(futures):
                entity, description, error = future.result()
                descriptions[entity] = description
                completed += 1

                if completed % 2 == 0 or completed == len(entities):
                    elapsed = time.time() - start_time
                    logger.info(f"  进度: {completed}/{len(entities)} ({elapsed:.2f}s)")

        elapsed_time = time.time() - start_time
        successful = len([d for d in descriptions.values() if d])

        logger.info(
            f"[Sentence.generate_affective_descriptions_parallel] 并行生成完成"
            f" ({elapsed_time:.2f}s, {successful}/{len(entities)} 成功)"
        )

        return descriptions

    def generate_affective_descriptions_batch(
        self,
        sentence: str,
        entities: List[str]
    ) -> Dict[str, str]:
        """
        批量版本：一次性生成所有实体的情感描述

        优化方案六A：使用批量 Prompt，将 N 次 LLM 调用减少到 1 次

        Args:
            sentence: 原始句子 S
            entities: 实体列表 E = {e_1, e_2, ...}

        Returns:
            dict: 实体到情感描述的映射 {e_1: d_1, e_2: d_2, ...}
        """
        if not entities:
            return {}

        start_time = time.time()

        logger.info(f"[Sentence.generate_affective_descriptions_batch] 开始批量生成情感描述")
        logger.info(f"  实体数量: {len(entities)}")

        # 准备实体列表格式
        entities_list = "\n".join([f"- {entity}" for entity in entities])

        # 使用批量模板渲染 prompt
        try:
            prompt = self.prompt_template_manager.render(
                name='affective_description_batch',
                sentence=sentence,
                entities_list=entities_list
            )
        except Exception as e:
            logger.warning(f"批量模板未找到，回退到并行模式: {e}")
            return self.generate_affective_descriptions_parallel(sentence, entities)

        try:
            logger.debug(f"[Batch] Prompt: {prompt[:500]}{'...' if len(prompt) > 500 else ''}")

            result = self.client.complete(
                query=prompt,
                max_tokens=200,  # 增加以适应多个实体
                temperature=0.7
            )

            response_text = result.get_answer_text().strip()

            logger.info(f"[Batch] 原始响应: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")

            # 解析批量响应
            descriptions = self._parse_batch_response(response_text, entities)

            elapsed_time = time.time() - start_time
            successful = len([d for d in descriptions.values() if d])

            logger.info(
                f"[Sentence.generate_affective_descriptions_batch] 批量生成完成"
                f" ({elapsed_time:.2f}s, {successful}/{len(entities)} 成功)"
            )

            return descriptions

        except Exception as e:
            logger.error(f"[Sentence.generate_affective_descriptions_batch] 批量生成失败")
            logger.error(f"  错误信息: {str(e)}")
            import traceback
            logger.error(f"  错误堆栈:\n{traceback.format_exc()}")

            # 失败时回退到并行模式
            logger.warning("回退到并行模式...")
            return self.generate_affective_descriptions_parallel(sentence, entities)

    def _parse_batch_response(
        self,
        response_text: str,
        entities: List[str]
    ) -> Dict[str, str]:
        """
        解析批量响应，提取每个实体的情感描述

        Args:
            response_text: LLM 返回的批量响应
            entities: 实体列表

        Returns:
            dict: 实体到情感描述的映射
        """
        descriptions = {entity: "" for entity in entities}

        # 分割响应（按行或按实体标记）
        lines = response_text.split('\n')

        for line in lines:
            line = line.strip()

            # 尝试匹配 "Entity: description" 格式
            for entity in entities:
                # 尝试不同的匹配模式
                patterns = [
                    f"{entity}:",
                    f"- {entity}:",
                    f"{entity} -",
                ]

                matched = False
                for pattern in patterns:
                    if pattern in line:
                        # 提取描述部分
                        desc_part = line.split(pattern, 1)[1].strip()
                        # 清理描述（移除行号、多余标点等）
                        desc_part = self._clean_description(desc_part)

                        if desc_part and not descriptions[entity]:
                            descriptions[entity] = desc_part
                            logger.debug(f"[Batch] 解析到实体 '{entity}': {desc_part}")
                            matched = True
                            break

                if matched:
                    break

        # 如果解析失败，尝试宽松匹配（实体名称出现在某一行）
        for entity in entities:
            if not descriptions[entity]:
                for line in lines:
                    if entity.lower() in line.lower():
                        # 提取可能的情感词
                        words = line.split()
                        emotion_words = []
                        for word in words:
                            word = word.strip(',.:-;')
                            if len(word) > 2 and word.lower() not in ['the', 'and', 'for', 'are']:
                                emotion_words.append(word)

                        if emotion_words and not descriptions[entity]:
                            descriptions[entity] = ", ".join(emotion_words[:8])
                            logger.debug(f"[Batch] 宽松匹配实体 '{entity}': {descriptions[entity]}")
                            break

        return descriptions

    def _clean_description(self, desc: str) -> str:
        """
        清理情感描述文本

        Args:
            desc: 原始描述

        Returns:
            str: 清理后的描述
        """
        # 移除行号（如 "1. ", "2. "）
        desc = desc.strip()

        # 移除开头的序号
        import re
        desc = re.sub(r'^[\d]+[\.\)]\s*', '', desc)

        # 移除多余的空格和标点
        desc = desc.strip(',.:-; ')

        return desc

