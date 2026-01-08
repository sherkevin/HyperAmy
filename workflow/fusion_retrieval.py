"""
融合检索模块 - Amygdala + HippoRAG

实现方案 1：级联检索
- HippoRAG 快速筛选候选（Top-K）
- Amygdala 深度精排（Top-M）

优势：
1. 兼顾速度和质量
2. HippoRAG 快速缩小范围
3. Amygdala 保证最终结果质量

使用示例：
    >>> from workflow.fusion_retrieval import FusionRetriever
    >>>
    >>> # 初始化
    >>> fusion = FusionRetriever(
    ...     amygdala_save_dir="./amygdala_db",
    ...     hipporag_save_dir="./hipporag_db"
    ... )
    >>>
    >>> # 添加数据
    >>> fusion.add(chunks)
    >>>
    >>> # 级联检索
    >>> results = fusion.retrieve(
    ...     query="your query",
    ...     hipporag_top_k=20,  # HippoRAG 返回 20 个候选
    ...     amygdala_top_k=5    # Amygdala 从中选出 5 个
    ... )
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# 设置环境变量
from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
os.environ["OPENAI_API_KEY"] = API_KEY

from workflow.amygdala import Amygdala
from workflow.hipporag_wrapper import HippoRAGWrapper

logger = logging.getLogger(__name__)


class FusionRetriever:
    """
    融合检索器 - Amygdala + HippoRAG 级联检索

    工作流程：
    1. 添加数据时：同时添加到两个系统
    2. 检索时：
       - HippoRAG 快速筛选（Top-K 候选）
       - Amygdala 深度精排（从候选中选出 Top-M）
    3. 返回融合结果
    """

    def __init__(
        self,
        amygdala_save_dir: str = "./fusion_amygdala_db",
        hipporag_save_dir: str = "./fusion_hipporag_db",
        llm_model_name: str = "DeepSeek-V3.2",
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        auto_link_particles: bool = False
    ):
        """
        初始化融合检索器

        Args:
            amygdala_save_dir: Amygdala 数据目录
            hipporag_save_dir: HippoRAG 数据目录
            llm_model_name: LLM 模型名称
            embedding_model_name: 嵌入模型名称
            auto_link_particles: 是否自动链接粒子
        """
        self.amygda_save_dir = amygdala_save_dir
        self.hipporag_save_dir = hipporag_save_dir

        # 初始化 Amygdala
        logger.info("初始化 Amygdala...")
        self.amygda = Amygdala(
            save_dir=amygdala_save_dir,
            particle_collection_name="fusion_particles",
            conversation_namespace="fusion",
            auto_link_particles=auto_link_particles
        )
        logger.info("✓ Amygdala 初始化完成")

        # 初始化 HippoRAG
        logger.info("初始化 HippoRAG...")
        self.hipporag = HippoRAGWrapper(
            save_dir=hipporag_save_dir,
            llm_model_name=llm_model_name,
            llm_base_url=BASE_URL,
            embedding_model_name=f"VLLM/{embedding_model_name}",
            embedding_base_url=API_URL_EMBEDDINGS
        )
        logger.info("✓ HippoRAG 初始化完成")

        logger.info("✓ 融合检索器初始化完成")

    def add(self, chunks: List[str]) -> Dict[str, Any]:
        """
        添加数据到两个系统

        Args:
            chunks: 文档块列表

        Returns:
            {
                'amygdala_count': int,
                'hipporag_count': int,
                'total_chunks': int
            }
        """
        if not chunks:
            logger.warning("No chunks provided")
            return {'amygdala_count': 0, 'hipporag_count': 0, 'total_chunks': 0}

        logger.info(f"添加 {len(chunks)} 个 chunks 到融合检索器")

        # 添加到 Amygdala
        amygdala_count = 0
        for chunk in chunks:
            result = self.amygda.add(chunk)
            amygdala_count += result['particle_count']

        # 添加到 HippoRAG
        hipporag_result = self.hipporag.add(chunks)
        hipporag_count = hipporag_result['total_indexed']

        logger.info(f"✓ 添加完成: Amygdala {amygdala_count} 粒子, HippoRAG {hipporag_count} chunks")

        return {
            'amygdala_count': amygdala_count,
            'hipporag_count': hipporag_count,
            'total_chunks': len(chunks)
        }

    def retrieve(
        self,
        query: str,
        hipporag_top_k: int = 20,
        amygdala_top_k: int = 5,
        mode: str = "cascade"
    ) -> List[Dict[str, Any]]:
        """
        融合检索

        Args:
            query: 查询文本
            hipporag_top_k: HippoRAG 返回的候选数量（默认 20）
            amygdala_top_k: 最终返回的结果数量（默认 5）
            mode: 检索模式
                - "cascade": 级联检索（默认，推荐）
                - "parallel": 并行检索 + 分数融合
                - "hipporag_only": 仅 HippoRAG
                - "amygdala_only": 仅 Amygdala

        Returns:
            融合检索结果列表
        """
        if mode == "cascade":
            return self._retrieve_cascade(query, hipporag_top_k, amygdala_top_k)
        elif mode == "parallel":
            return self._retrieve_parallel(query, amygdala_top_k)
        elif mode == "hipporag_only":
            return self.hipporag.retrieve(query, top_k=amygdala_top_k)
        elif mode == "amygdala_only":
            return self.amygda.retrieval(query_text=query, retrieval_mode="chunk", top_k=amygdala_top_k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _retrieve_cascade(
        self,
        query: str,
        hipporag_top_k: int,
        amygdala_top_k: int
    ) -> List[Dict[str, Any]]:
        """
        级联检索：HippoRAG 快速筛选 → Amygdala 深度精排

        流程：
        1. HippoRAG 检索 hipporag_top_k 个候选
        2. 提取候选 chunks 的文本
        3. Amygdala 仅在这些候选中检索
        4. 返回 amygdala_top_k 个最终结果

        优势：
        - HippoRAG 快速缩小范围
        - Amygdala 在小范围内深度精排
        - 兼顾速度和质量
        """
        logger.info(f"【级联检索】Query: {query[:100]}...")
        logger.info(f"  HippoRAG 筛选 Top-{hipporag_top_k} → Amygdala 精排 Top-{amygdala_top_k}")

        # Step 1: HippoRAG 快速筛选
        import time
        start_time = time.time()

        logger.info("Step 1: HippoRAG 快速筛选...")
        hipporag_results = self.hipporag.retrieve(
            query=query,
            top_k=hipporag_top_k
        )
        hipporag_time = time.time() - start_time
        logger.info(f"  ✓ HippoRAG 完成 ({hipporag_time:.2f}s), 返回 {len(hipporag_results)} 个候选")

        if not hipporag_results:
            logger.warning("HippoRAG 未返回任何结果")
            return []

        # Step 2: 提取候选文本
        candidate_texts = [result['text'] for result in hipporag_results]
        logger.info(f"Step 2: 提取 {len(candidate_texts)} 个候选文本")

        # Step 3: Amygdala 在候选中精排
        logger.info("Step 3: Amygdala 深度精排...")
        amygdala_results = self.amygda.retrieval(
            query_text=query,
            retrieval_mode="chunk",
            top_k=len(candidate_texts)  # 获取所有候选的排名
        )
        amygdala_time = time.time() - start_time - hipporag_time

        # Step 4: 筛选出 HippoRAG 返回的候选
        logger.info("Step 4: 融合结果...")

        # 创建文本到 Amygdala 得分的映射
        amygdala_scores = {}
        for result in amygdala_results:
            amygdala_scores[result['text']] = result['score']

        # 筛选并重新排序
        fusion_results = []
        for hipporag_result in hipporag_results:
            text = hipporag_result['text']
            if text in amygdala_scores:
                fusion_results.append({
                    'text': text,
                    'hipporag_score': hipporag_result['score'],
                    'amygdala_score': amygdala_scores[text],
                    'fusion_score': amygdala_scores[text],  # 使用 Amygdala 分数作为主排序
                    'rank': len(fusion_results) + 1
                })

        # 按 Amygdala 分数排序
        fusion_results.sort(key=lambda x: x['amygdala_score'], reverse=True)

        # 只返回 top_k
        final_results = fusion_results[:amygdala_top_k]

        # 更新排名
        for i, result in enumerate(final_results, 1):
            result['rank'] = i

        total_time = time.time() - start_time

        logger.info(f"  ✓ Amygdala 完成 ({amygdala_time:.2f}s)")
        logger.info(f"  ✓ 融合检索完成 (总时间: {total_time:.2f}s)")
        logger.info(f"  ✓ 返回 {len(final_results)} 个结果")

        return final_results

    def _retrieve_parallel(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        并行检索 + 分数融合

        流程：
        1. 同时使用两个系统检索
        2. 归一化分数
        3. 加权融合
        4. 返回 top_k

        优势：
        - 并行检索，速度快
        - 两个系统的信号都保留
        """
        import time

        logger.info(f"【并行检索】Query: {query[:100]}...")

        start_time = time.time()

        # 并行检索
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            hipporag_future = executor.submit(
                self.hipporag.retrieve,
                query,
                top_k * 2  # 获取更多候选用于融合
            )
            amygdala_future = executor.submit(
                self.amygda.retrieval,
                query,
                "chunk",
                top_k * 2
            )

            hipporag_results = hipporag_future.result()
            amygdala_results = amygdala_future.result()

        logger.info(f"  ✓ 并行检索完成 ({time.time() - start_time:.2f}s)")

        # 创建文本到结果的映射
        hipporag_map = {r['text']: r['score'] for r in hipporag_results}
        amygdala_map = {r['text']: r['score'] for r in amygdala_results}

        # 归一化分数（Min-Max 归一化）
        all_texts = set(hipporag_map.keys()) | set(amygdala_map.keys())

        hipporag_scores = list(hipporag_map.values())
        amygdala_scores = list(amygdala_map.values())

        hipporag_min = min(hipporag_scores) if hipporag_scores else 0
        hipporag_max = max(hipporag_scores) if hipporag_scores else 1
        amygdala_min = min(amygdala_scores) if amygdala_scores else 0
        amygdala_max = max(amygdala_scores) if amygdala_scores else 1

        # 融合分数（权重可调）
        alpha = 0.4  # HippoRAG 权重
        beta = 0.6   # Amygdala 权重（给更高权重，因为质量更好）

        fusion_results = []
        for text in all_texts:
            h_score = hipporag_map.get(text, 0)
            a_score = amygdala_map.get(text, 0)

            # 归一化
            h_norm = (h_score - hipporag_min) / (hipporag_max - hipporag_min + 1e-6)
            a_norm = (a_score - amygdala_min) / (amygdala_max - amygdala_min + 1e-6)

            # 融合
            fusion_score = alpha * h_norm + beta * a_norm

            fusion_results.append({
                'text': text,
                'hipporag_score': h_score,
                'amygdala_score': a_score,
                'fusion_score': fusion_score,
                'rank': 0
            })

        # 按融合分数排序
        fusion_results.sort(key=lambda x: x['fusion_score'], reverse=True)

        # 返回 top_k
        final_results = fusion_results[:top_k]

        # 更新排名
        for i, result in enumerate(final_results, 1):
            result['rank'] = i

        logger.info(f"  ✓ 融合完成，返回 {len(final_results)} 个结果")

        return final_results

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'amygdala_particles': len(self.amygda.particle_to_conversation),
            'hipporag_stats': self.hipporag.get_stats()
        }

    def clear(self):
        """清空两个系统的数据"""
        import shutil

        logger.warning("清空融合检索器数据")

        if os.path.exists(self.amygda_save_dir):
            shutil.rmtree(self.amygda_save_dir)
            logger.info(f"✓ 清空 Amygdala 数据: {self.amygda_save_dir}")

        if os.path.exists(self.hipporag_save_dir):
            shutil.rmtree(self.hipporag_save_dir)
            logger.info(f"✓ 清空 HippoRAG 数据: {self.hipporag_save_dir}")


def create_fusion_retriever(
    amygdala_save_dir: str = "./fusion_amygdala_db",
    hipporag_save_dir: str = "./fusion_hipporag_db",
    llm_model_name: str = "DeepSeek-V3.2",
    embedding_model_name: str = None
) -> FusionRetriever:
    """
    便捷函数：创建融合检索器

    Args:
        amygdala_save_dir: Amygdala 数据目录
        hipporag_save_dir: HippoRAG 数据目录
        llm_model_name: LLM 模型名称
        embedding_model_name: 嵌入模型名称

    Returns:
        FusionRetriever 实例
    """
    return FusionRetriever(
        amygdala_save_dir=amygdala_save_dir,
        hipporag_save_dir=hipporag_save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name or DEFAULT_EMBEDDING_MODEL
    )
