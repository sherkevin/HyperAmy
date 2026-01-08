"""
HippoRAG Wrapper - 简洁的 HippoRAG 接口

这个模块提供了一个简洁的接口来使用 HippoRAG，隐藏了复杂的内部实现细节。

核心功能：
1. add(chunks) - 添加文档块到索引
2. retrieve(query, top_k) - 检索相关文档块

使用示例：
    >>> from workflow.hipporag_wrapper import HippoRAGWrapper
    >>>
    >>> # 初始化
    >>> wrapper = HippoRAGWrapper(
    ...     save_dir="./my_hipporag_db",
    ...     llm_model_name="DeepSeek-V3.2",
    ...     embedding_model_name="GLM-Embedding-2"
    ... )
    >>>
    >>> # 添加文档
    >>> chunks = [
    ...     "Python is a high-level programming language.",
    ...     "JavaScript is used for web development.",
    ...     "Java is widely used in enterprise applications."
    ... ]
    >>> wrapper.add(chunks)
    >>>
    >>> # 检索
    >>> results = wrapper.retrieve(
    ...     query="What programming languages are mentioned?",
    ...     top_k=2
    ... )
    >>>
    >>> for i, result in enumerate(results, 1):
    ...     print(f"Rank {i}: {result['text']}")
    ...     print(f"Score: {result['score']:.4f}")
    ...     print()
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# HippoRAG imports
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Logging setup
logger = logging.getLogger(__name__)


class HippoRAGWrapper:
    """
    HippoRAG 的简洁包装类

    提供简化的接口来使用 HippoRAG 的核心功能：
    - 添加文档块到索引
    - 检索相关文档块

    HippoRAG 工作原理：
    1. **索引阶段**：
       - 文档分块存储
       - OpenIE 提取实体和三元组
       - 构建知识图谱
       - 编码实体和事实的嵌入

    2. **检索阶段**：
       - 查询编码（query_to_fact, query_to_passage）
       - 事实检索和重排序
       - 图谱搜索（PPR）
       - 返回 top-k 相关文档
    """

    def __init__(
        self,
        save_dir: str = "./hipporag_db",
        llm_model_name: str = None,
        embedding_model_name: str = None,
        llm_base_url: str = None,
        embedding_base_url: str = None,
        **kwargs
    ):
        """
        初始化 HippoRAG 包装器

        Args:
            save_dir: 数据库保存目录
            llm_model_name: LLM 模型名称（如 "DeepSeek-V3.2"）
            embedding_model_name: 嵌入模型名称（如 "GLM-Embedding-2"）
            llm_base_url: LLM 服务 URL
            embedding_base_url: 嵌入服务 URL
            **kwargs: 其他 HippoRAG 配置参数
        """
        self.save_dir = save_dir

        # 使用默认配置
        self.global_config = BaseConfig(
            save_dir=save_dir,
            llm_name=llm_model_name or "DeepSeek-V3.2",
            embedding_model_name=embedding_model_name or "GLM-Embedding-2",
        )

        # 覆盖 URL 配置
        if llm_base_url:
            self.global_config.llm_base_url = llm_base_url
        if embedding_base_url:
            self.global_config.embedding_base_url = embedding_base_url

        # 覆盖额外配置
        for key, value in kwargs.items():
            if hasattr(self.global_config, key):
                setattr(self.global_config, key, value)

        # 初始化 HippoRAG
        logger.info(f"Initializing HippoRAG wrapper with save_dir: {save_dir}")
        self.hipporag = HippoRAG(global_config=self.global_config)

        # 统计信息
        self._indexed_count = 0
        self._chunk_ids = []

        logger.info("✓ HippoRAG wrapper initialized")

    def add(self, chunks: List[str]) -> Dict[str, Any]:
        """
        添加文档块到索引

        Args:
            chunks: 文档块列表

        Returns:
            包含添加信息的字典：
            {
                'chunk_count': int,      # 添加的块数量
                'chunk_ids': List[str],  # 块 ID 列表
                'total_indexed': int     # 总索引块数
            }

        流程：
        1. 将 chunks 存入 chunk_embedding_store
        2. 执行 OpenIE 提取实体和三元组
        3. 编码实体和事实的嵌入
        4. 构建图谱（事实边 + 文档边 + 同义边）
        5. 保存图谱
        """
        if not chunks:
            logger.warning("No chunks provided, skipping add operation")
            return {'chunk_count': 0, 'chunk_ids': [], 'total_indexed': self._indexed_count}

        logger.info(f"Adding {len(chunks)} chunks to HippoRAG index")

        # 调用 HippoRAG 的 index 方法
        self.hipporag.index(docs=chunks)

        # 更新统计信息
        self._indexed_count += len(chunks)
        chunk_ids = [self.hipporag.chunk_embedding_store.text_to_hash_id[chunk] for chunk in chunks]
        self._chunk_ids.extend(chunk_ids)

        logger.info(f"✓ Indexed {len(chunks)} chunks, total indexed: {self._indexed_count}")

        return {
            'chunk_count': len(chunks),
            'chunk_ids': chunk_ids,
            'total_indexed': self._indexed_count
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档块

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认 5）
            return_scores: 是否返回得分（默认 True）

        Returns:
            检索结果列表，每个结果包含：
            {
                'text': str,          # 文档文本
                'score': float,       # 相关性得分（如果 return_scores=True）
                'rank': int           # 排名
            }

        检索流程：
        1. 编码查询（query_to_fact, query_to_passage）
        2. 检索相关事实并重排序
        3. 图谱搜索（PPR）
        4. 返回 top-k 相关文档
        """
        if not query:
            logger.warning("Empty query provided")
            return []

        logger.info(f"Retrieving top-{top_k} chunks for query: {query[:100]}...")

        # 调用 HippoRAG 的 retrieve 方法
        query_solutions = self.hipporag.retrieve(
            queries=[query],
            num_to_retrieve=top_k
        )

        # 解析结果
        results = []
        query_solution = query_solutions[0]  # 只有一个查询

        for rank, (doc, score) in enumerate(zip(query_solution.docs, query_solution.doc_scores), 1):
            result = {
                'text': doc,
                'rank': rank
            }
            if return_scores:
                result['score'] = float(score)
            results.append(result)

        logger.info(f"✓ Retrieved {len(results)} chunks")

        return results

    def retrieve_dpr(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        使用 DPR（密集检索）检索相关文档块

        这是一个更简单但精度较低的检索方法，不使用图谱。

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认 5）

        Returns:
            检索结果列表
        """
        logger.info(f"Retrieving with DPR (no graph) for query: {query[:100]}...")

        # 调用 HippoRAG 的 retrieve_dpr 方法
        query_solutions = self.hipporag.retrieve_dpr(
            queries=[query],
            num_to_retrieve=top_k
        )

        # 解析结果
        results = []
        query_solution = query_solutions[0]

        for rank, (doc, score) in enumerate(zip(query_solution.docs, query_solution.doc_scores), 1):
            results.append({
                'text': doc,
                'score': float(score),
                'rank': rank
            })

        logger.info(f"✓ Retrieved {len(results)} chunks with DPR")

        return results

    def qa(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        执行 RAG 问答（检索 + 生成）

        Args:
            query: 查询问题
            top_k: 检索文档数量（默认 5）

        Returns:
            {
                'answer': str,              # 生成的回答
                'retrieved_chunks': List,   # 检索到的文档块
                'messages': List,           # LLM 对话消息
                'metadata': Dict            # 元数据
            }
        """
        logger.info(f"Performing RAG QA for query: {query[:100]}...")

        # 调用 HippoRAG 的 rag_qa 方法
        query_solutions, messages, metadata = self.hipporag.rag_qa(queries=[query])

        query_solution = query_solutions[0]

        # 提取检索到的文档块
        retrieved_chunks = []
        if hasattr(query_solution, 'docs') and query_solution.docs:
            for rank, (doc, score) in enumerate(zip(query_solution.docs, query_solution.doc_scores), 1):
                retrieved_chunks.append({
                    'text': doc,
                    'score': float(score),
                    'rank': rank
                })

        result = {
            'answer': query_solution.answer,
            'retrieved_chunks': retrieved_chunks,
            'messages': messages,
            'metadata': metadata[0] if metadata else {}
        }

        logger.info(f"✓ Generated answer: {result['answer'][:100]}...")

        return result

    def delete(self, chunks: List[str]) -> Dict[str, Any]:
        """
        删除文档块及其相关内容

        Args:
            chunks: 要删除的文档块列表

        Returns:
            {
                'deleted_count': int,  # 删除的块数量
                'remaining_count': int # 剩余的块数量
            }
        """
        if not chunks:
            logger.warning("No chunks provided for deletion")
            return {'deleted_count': 0, 'remaining_count': self._indexed_count}

        logger.info(f"Deleting {len(chunks)} chunks from index")

        # 调用 HippoRAG 的 delete 方法
        self.hipporag.delete(docs_to_delete=chunks)

        # 更新统计
        self._indexed_count -= len(chunks)

        logger.info(f"✓ Deleted {len(chunks)} chunks, remaining: {self._indexed_count}")

        return {
            'deleted_count': len(chunks),
            'remaining_count': self._indexed_count
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息

        Returns:
            {
                'total_indexed': int,        # 总索引块数
                'graph_nodes': int,          # 图谱节点数
                'graph_edges': int,          # 图谱边数
                'entities': int,             # 实体数量
                'facts': int                 # 事实数量
            }
        """
        stats = {
            'total_indexed': self._indexed_count,
            'graph_nodes': self.hipporag.graph.vcount(),
            'graph_edges': self.hipporag.graph.ecount(),
            'entities': len(self.hipporag.entity_embedding_store.hash_ids),
            'facts': len(self.hipporag.fact_embedding_store.hash_ids)
        }

        return stats

    def clear(self):
        """
        清空索引（删除所有数据）

        警告：此操作不可逆！
        """
        logger.warning(f"Clearing all HippoRAG data in {self.save_dir}")

        # 删除目录
        import shutil
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            logger.info(f"✓ Cleared HippoRAG data directory: {self.save_dir}")

        # 重置统计
        self._indexed_count = 0
        self._chunk_ids = []

        logger.info("✓ Index cleared")


def create_hipporag_wrapper(
    save_dir: str = "./hipporag_db",
    llm_model_name: str = None,
    embedding_model_name: str = None,
    llm_base_url: str = None,
    embedding_base_url: str = None
) -> HippoRAGWrapper:
    """
    便捷函数：创建 HippoRAG 包装器实例

    Args:
        save_dir: 数据库保存目录
        llm_model_name: LLM 模型名称
        embedding_model_name: 嵌入模型名称
        llm_base_url: LLM 服务 URL
        embedding_base_url: 嵌入服务 URL

    Returns:
        HippoRAGWrapper 实例
    """
    return HippoRAGWrapper(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=llm_base_url,
        embedding_base_url=embedding_base_url
    )
