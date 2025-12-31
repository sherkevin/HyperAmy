"""
Emotion V2 类

使用 HippoRAG 的实体抽取功能
"""
import os

from hipporag.information_extraction import OpenIE
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.logging_utils import get_logger
from llm.config import API_KEY, API_URL_CHAT, DEFAULT_MODEL

logger = get_logger(__name__)


class Entity:
    """
    实体抽取类
    
    使用 HippoRAG 的 OpenIE 模块来抽取文本中的实体
    """
    
    def __init__(self, model_name=None, base_url=None, cache_dir=None):
        """
        初始化 Entity 类
        
        Args:
            model_name: 使用的模型名称，如果为 None 则使用默认模型
            base_url: API 基础 URL，如果为 None 则使用默认 URL
            cache_dir: 缓存目录，如果为 None 则使用临时目录
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.base_url = base_url or API_URL_CHAT.replace('/chat/completions', '')
        
        # 设置环境变量（HippoRAG 需要）
        os.environ['OPENAI_API_KEY'] = API_KEY
        
        # 创建配置
        self.config = BaseConfig()
        self.config.llm_name = self.model_name
        self.config.llm_base_url = self.base_url
        
        # 设置缓存目录和保存目录
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache', 'llm_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 设置 save_dir（CacheOpenAI.from_experiment_config 需要）
        # BaseConfig 默认 save_dir 为 None，需要设置为实际目录
        save_dir = os.path.dirname(cache_dir) if os.path.dirname(cache_dir) else os.path.join(os.path.dirname(__file__), '..', '.cache')
        self.config.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化 LLM 模型
        self.llm_model = CacheOpenAI.from_experiment_config(self.config)
        
        # 初始化 OpenIE
        self.openie = OpenIE(llm_model=self.llm_model)
        
        logger.info(f"Entity initialized with model: {self.model_name}")
    
    def extract_entities(self, chunk: str) -> list:
        """
        从 chunk 中提取命名实体

        Args:
            chunk: 输入文本片段

        Returns:
            list: 命名实体列表
        """
        # 检查空文本
        if not chunk or not chunk.strip():
            logger.debug("Input chunk is empty, returning empty entity list")
            return []

        try:
            # 使用 OpenIE 的 NER 功能
            ner_output = self.openie.ner(chunk_key="temp", passage=chunk)

            # 返回唯一实体列表
            entities = ner_output.unique_entities

            logger.debug(f"Extracted {len(entities)} entities from chunk: {entities}")

            return entities

        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            raise
    
    def extract_triples(self, chunk: str, named_entities: list = None) -> list:
        """
        从 chunk 中提取三元组（实体-关系-实体）
        
        Args:
            chunk: 输入文本片段
            named_entities: 命名实体列表，如果为 None 则先提取实体
        
        Returns:
            list: 三元组列表，每个三元组格式为 [subject, relation, object]
        """
        try:
            # 如果没有提供实体，先提取实体
            if named_entities is None:
                ner_output = self.openie.ner(chunk_key="temp", passage=chunk)
                named_entities = ner_output.unique_entities
            
            # 使用 OpenIE 的三元组提取功能
            triple_output = self.openie.triple_extraction(
                chunk_key="temp",
                passage=chunk,
                named_entities=named_entities
            )
            
            # 返回三元组列表
            triples = triple_output.triples
            
            logger.debug(f"Extracted {len(triples)} triples from chunk")
            
            return triples
            
        except Exception as e:
            logger.error(f"Failed to extract triples: {e}")
            raise
    
    def extract_all(self, chunk: str) -> dict:
        """
        从 chunk 中提取实体和三元组
        
        Args:
            chunk: 输入文本片段
        
        Returns:
            dict: 包含 'entities' 和 'triples' 的字典
        """
        try:
            # 使用 OpenIE 的完整功能
            result = self.openie.openie(chunk_key="temp", passage=chunk)
            
            entities = result["ner"].unique_entities
            triples = result["triplets"].triples
            
            logger.debug(
                f"Extracted {len(entities)} entities and {len(triples)} triples from chunk"
            )
            
            return {
                "entities": entities,
                "triples": triples
            }
            
        except Exception as e:
            logger.error(f"Failed to extract entities and triples: {e}")
            raise

