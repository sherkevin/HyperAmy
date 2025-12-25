"""
Sentence 类

实现 Step 2: 为每个实体生成情感视角描述 (Affective Perspective Description)
"""
from typing import List, Dict
from llm import create_client
from llm.config import API_URL_CHAT, DEFAULT_MODEL
from hipporag.utils.logging_utils import get_logger
from prompts.prompt_template_manager import PromptTemplateManager

logger = get_logger(__name__)


class Sentence:
    """
    句子处理类
    
    功能：
    - Step 1: 实体识别（使用 Entity 类）
    - Step 2: 为每个实体生成情感视角描述
    """
    
    def __init__(self, model_name=None):
        """
        初始化 Sentence 类
        
        Args:
            model_name: 使用的模型名称，如果为 None 则使用默认模型
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
        
        logger.info(f"Sentence initialized with model: {self.model_name}")
    
    def generate_affective_description(self, sentence: str, entity: str) -> str:
        """
        为单个实体生成情感视角描述
        
        Prompt 逻辑：基于句子 S，生成关于 e_i 的纯情绪化描述。
        忽略事实细节，只保留情感色彩。
        
        Args:
            sentence: 原始句子 S
            entity: 实体 e_i
        
        Returns:
            str: 情感视角描述 d_i
        """
        # 使用模板管理器渲染 prompt
        prompt = self.prompt_template_manager.render(
            name='affective_description',
            sentence=sentence,
            entity=entity
        )

        try:
            result = self.client.complete(
                query=prompt,
                max_tokens=50,  # 情绪词列表较短，减少 token 数
                temperature=0.7  # 稍高温度以增加描述的多样性
            )
            
            description = result.get_answer_text().strip()
            
            logger.debug(
                f"Generated affective description for entity '{entity}': {description[:50]}..."
            )
            
            return description
            
        except Exception as e:
            logger.error(f"Failed to generate affective description for entity '{entity}': {e}")
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

