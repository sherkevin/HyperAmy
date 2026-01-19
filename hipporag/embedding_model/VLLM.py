from typing import List
import os
import numpy as np
from tqdm import tqdm

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction
import requests

class VLLMEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name starts with "VLLM/"
    The embedding base url should contain the v1/embeddings.
    """
    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name[len("VLLM/"):]
        self.embedding_type = 'float'
        self.batch_size = 16  # 减小batch size避免"Request Entity Too Large"错误

        self.url = global_config.embedding_base_url
        self.base_url = global_config.embedding_base_url

        self.search_query_instr = set([
            get_query_instruction('query_to_fact'),
            get_query_instruction('query_to_passage')
        ])

    def call_model(self, input_text) -> List[np.ndarray]:
        if isinstance(input_text, str):
            input_text = [input_text]
        
        original_count = len(input_text)
        
        # 过滤空字符串和None，但记录原始索引
        filtered_texts = []
        original_indices = []
        for idx, text in enumerate(input_text):
            if text and isinstance(text, str) and text.strip():
                filtered_texts.append(text)
                original_indices.append(idx)
        
        if not filtered_texts:
            # 如果所有文本都为空，返回对应数量的零向量（保持长度一致）
            return np.zeros((original_count, 2048))  # GLM-Embedding-3的维度是2048
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"
        }
        
        payload = {
            "model": self.model_id,
            "input": filtered_texts,  # 使用过滤后的文本
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # 获取embeddings
            embeddings_list = [result["data"][i]["embedding"] for i in range(len(result["data"]))]
            filtered_embeddings = np.array(embeddings_list)
            
            # 如果过滤掉了某些文本，需要在对应位置插入零向量
            if len(filtered_embeddings) < original_count:
                # 创建完整的结果数组
                embed_dim = filtered_embeddings.shape[1] if len(filtered_embeddings.shape) > 1 else 2048
                full_embeddings = np.zeros((original_count, embed_dim))
                for i, orig_idx in enumerate(original_indices):
                    full_embeddings[orig_idx] = filtered_embeddings[i]
                return full_embeddings
            
            return filtered_embeddings
        except requests.exceptions.HTTPError as e:
            # 添加更详细的错误信息
            if hasattr(e.response, 'text'):
                error_detail = e.response.text[:500]
                print(f"API Error: {e}")
                print(f"Request URL: {self.base_url}")
                print(f"Request payload keys: {list(payload.keys())}")
                print(f"Input count: {len(input_text)}")
                print(f"Error response: {error_detail}")
            raise

    def encode(self, texts: List[str]) -> np.array:
        response = self.call_model(texts)
        return response

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if len(texts) < self.batch_size:
            return self.encode(texts)
        
        results = []
        batch_indexes = list(range(0, len(texts), self.batch_size))
        for i in tqdm(batch_indexes, desc="Batch Encoding"):
            results.append(self.encode(texts[i:i + self.batch_size]))
        return np.concatenate(results)
