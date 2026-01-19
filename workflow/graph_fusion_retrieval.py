"""
图谱融合检索 - HippoRAG + Amygdala 实体级融合

融合策略：
1. 统一的实体抽取（只调用一次 LLM）- 使用 UnifiedEntityExtractor
2. HippoRAG: 语义相似度扩展实体
3. Amygdala: 情绪相似度扩展实体
4. 将扩展的实体映射到 HippoRAG 实体空间
5. 融合权重 → PPR 传播
6. 返回排序后的 chunks

优势：
- 利用了 HippoRAG 的图谱结构（语义关系）
- 利用了 Amygdala 的情绪感知能力
- 在统一的 HippoRAG 图谱中进行 PPR 传播
- 保留了两个系统的信号

性能优化（V2）：
- 统一实体抽取，避免重复 LLM 调用
- 并行执行语义扩展、情绪扩展、fact 提取

使用示例：
    >>> from workflow.graph_fusion_retrieval import GraphFusionRetriever
    >>>
    >>> # 初始化
    >>> fusion = GraphFusionRetriever(
    ...     amygdala_save_dir="./amygdala_db",
    ...     hipporag_save_dir="./hipporag_db"
    ... )
    >>>
    >>> # 添加数据
    >>> fusion.add(chunks)
    >>>
    >>> # 融合检索
    >>> results = fusion.retrieve(
    ...     query="your query",
    ...     top_k=5,
    ...     emotion_weight=0.3  # Amygdala 权重
    ... )
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
os.environ["OPENAI_API_KEY"] = API_KEY

from workflow.amygdala import Amygdala
from workflow.hipporag_wrapper import HippoRAGWrapper
from workflow.unified_entity_extractor import get_global_extractor
from hipporag.utils.misc_utils import compute_mdhash_id
from hipporag.utils.embed_utils import retrieve_knn

logger = logging.getLogger(__name__)


class GraphFusionRetriever:
    """
    图谱融合检索器 - 实体级融合 HippoRAG + Amygdala

    工作流程：
    1. 添加数据时：同时添加到两个系统
    2. 检索时：
       a. 从 query 中抽取实体（统一抽取）
       b. HippoRAG 语义扩展 → semantic_entities
       c. Amygdala 情绪扩展 → emotion_entities
       d. 映射到 HippoRAG 实体空间
       e. 融合权重
       f. 运行 PPR 传播
       g. 返回排序后的 chunks
    """

    def __init__(
        self,
        amygdala_save_dir: str = "./graph_fusion_amygdala_db",
        hipporag_save_dir: str = "./graph_fusion_hipporag_db",
        llm_model_name: str = "DeepSeek-V3.2",
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        auto_link_particles: bool = False,
        enable_parallel: bool = True
    ):
        """
        初始化融合检索器

        Args:
            amygdala_save_dir: Amygdala 数据目录
            hipporag_save_dir: HippoRAG 数据目录
            llm_model_name: LLM 模型名称
            embedding_model_name: 嵌入模型名称
            auto_link_particles: 是否自动链接粒子
            enable_parallel: 是否启用并行化优化（默认 True）
        """
        self.amygda_save_dir = amygdala_save_dir
        self.hipporag_save_dir = hipporag_save_dir
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.enable_parallel = enable_parallel

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

        # 获取 HippoRAG 核心对象（用于内部访问）
        self._hipporag_core = self.hipporag.hipporag

        # 初始化统一实体抽取器（全局单例，避免重复 LLM 调用）
        logger.info("初始化统一实体抽取器...")
        self.entity_extractor = get_global_extractor()
        logger.info("✓ 统一实体抽取器初始化完成")

        logger.info(f"✓ 图谱融合检索器初始化完成 (并行化: {'启用' if enable_parallel else '禁用'})")

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
        top_k: int = 5,
        emotion_weight: float = 0.3,
        semantic_weight: float = 0.5,
        fact_weight: float = 0.2,
        linking_top_k: int = 20,
        passage_node_weight: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        融合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            emotion_weight: Amygdala 情绪权重（默认 0.3）
            semantic_weight: HippoRAG 语义权重（默认 0.5）
            fact_weight: HippoRAG fact 权重（默认 0.2）
            linking_top_k: HippoRAG 链接 top_k（默认 20）
            passage_node_weight: passage 节点权重（默认 0.05）

        Returns:
            融合检索结果列表
        """
        logger.info("=" * 80)
        logger.info(f"[GraphFusionRetriever.retrieve] 开始融合检索")
        logger.info(f"  Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        logger.info(f"  权重配置: emotion={emotion_weight}, semantic={semantic_weight}, fact={fact_weight}")
        logger.info(f"  并行模式: {'启用' if self.enable_parallel else '禁用'}")

        start_time = time.time()

        # 确保 HippoRAG 准备好检索
        if not self._hipporag_core.ready_to_retrieve:
            self._hipporag_core.prepare_retrieval_objects()

        # Step 1: 统一实体抽取（只调用一次 LLM）
        logger.info(f"[GraphFusionRetriever.retrieve] Step 1: 统一实体抽取（UnifiedEntityExtractor）")
        query_entities = self.entity_extractor.extract_entities(
            text=query,
            text_id=f"query_{int(time.time())}",
            use_llm=True
        )

        if not query_entities:
            logger.warning(f"[GraphFusionRetriever.retrieve] Query 未抽取到任何实体，使用 HippoRAG 直接检索")
            return self._fallback_to_hipporag(query, top_k)

        logger.info(f"  ✓ 统一抽取到 {len(query_entities)} 个实体: {query_entities}")

        # 根据并行模式执行 Steps 2-5
        if self.enable_parallel:
            semantic_entities, emotion_entities, fact_entities = self._parallel_expansion_steps(
                query=query,
                query_entities=query_entities,
                linking_top_k=linking_top_k
            )
        else:
            semantic_entities, emotion_entities, fact_entities = self._serial_expansion_steps(
                query=query,
                query_entities=query_entities,
                linking_top_k=linking_top_k
            )

        # Step 6: 融合实体权重
        logger.info(f"[GraphFusionRetriever.retrieve] Step 6: 融合实体权重")
        entity_weights = self._merge_entity_weights(
            query_entities=query_entities,
            semantic_entities=semantic_entities,
            emotion_entities=emotion_entities,
            fact_entities=fact_entities,
            emotion_weight=emotion_weight,
            semantic_weight=semantic_weight,
            fact_weight=fact_weight
        )
        logger.info(f"  ✓ 融合后共有 {len(entity_weights)} 个实体节点")

        # Step 7: 运行 PPR
        logger.info(f"[GraphFusionRetriever.retrieve] Step 7: 运行 PPR 传播")
        sorted_doc_ids, ppr_scores = self._run_ppr_with_entity_weights(
            entity_weights=entity_weights,
            passage_node_weight=passage_node_weight
        )
        logger.info(f"  ✓ PPR 完成")

        # Step 8: 格式化结果
        results = self._format_results(
            sorted_doc_ids=sorted_doc_ids,
            ppr_scores=ppr_scores,
            top_k=top_k
        )

        elapsed_time = time.time() - start_time
        logger.info(f"[GraphFusionRetriever.retrieve] 检索完成，耗时 {elapsed_time:.2f}s")
        logger.info("=" * 80)

        return results

    def _serial_expansion_steps(
        self,
        query: str,
        query_entities: List[str],
        linking_top_k: int
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        串行执行扩展步骤

        Args:
            query: 查询文本
            query_entities: 统一抽取的实体列表
            linking_top_k: 链接 top-k

        Returns:
            (semantic_entities, emotion_entities, fact_entities)
        """
        # Step 2: 获取 query embedding
        logger.info(f"[GraphFusionRetriever.retrieve] Step 2: 获取 query embedding")
        self._hipporag_core.get_query_embeddings([query])
        query_embedding = self._hipporag_core.query_to_embedding['triple'].get(query)
        logger.info(f"  ✓ Query embedding 准备完成")

        # Step 3: HippoRAG 语义扩展（使用统一抽取的实体）
        logger.info(f"[GraphFusionRetriever.retrieve] Step 3: HippoRAG 语义扩展")
        semantic_entities = self._semantic_expansion(
            query_entities=query_entities,
            top_k=linking_top_k
        )
        logger.info(f"  ✓ 语义扩展找到 {len(semantic_entities)} 个相似实体")

        # Step 4: Amygdala 情绪扩展（需要 emotion 向量，单独处理）
        logger.info(f"[GraphFusionRetriever.retrieve] Step 4: Amygdala 情绪扩展")
        emotion_entities = self._emotion_expansion_from_entities(
            query=query,
            query_entities=query_entities,
            top_k=linking_top_k
        )
        logger.info(f"  ✓ 情绪扩展找到 {len(emotion_entities)} 个实体")

        # Step 5: HippoRAG fact 实体
        logger.info(f"[GraphFusionRetriever.retrieve] Step 5: 提取 HippoRAG fact 实体")
        fact_entities = self._extract_fact_entities(
            query=query,
            top_k=linking_top_k
        )
        logger.info(f"  ✓ Fact 扩展找到 {len(fact_entities)} 个实体")

        return semantic_entities, emotion_entities, fact_entities

    def _parallel_expansion_steps(
        self,
        query: str,
        query_entities: List[str],
        linking_top_k: int
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        并行执行扩展步骤（性能优化）

        Steps 2-5 可以并行执行：
        - Step 2: query embedding
        - Step 3: 语义扩展（使用统一抽取的实体）
        - Step 4: 情绪扩展（需要 emotion 向量，单独处理）
        - Step 5: fact 提取

        Returns:
            (semantic_entities, emotion_entities, fact_entities)
        """
        logger.info(f"[GraphFusionRetriever.retrieve] Steps 2-5: 并行执行扩展任务")

        # 提交所有并行任务
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Task 1: 获取 query embedding
            future_embedding = executor.submit(
                self._get_query_embedding_task, query
            )

            # Task 2: 语义扩展（使用统一抽取的实体，不依赖 embedding）
            future_semantic = executor.submit(
                self._semantic_expansion,
                query_entities=query_entities,
                top_k=linking_top_k
            )

            # Task 3: 情绪扩展（需要 emotion 向量，从 entities 生成）
            future_emotion = executor.submit(
                self._emotion_expansion_from_entities,
                query=query,
                query_entities=query_entities,
                top_k=linking_top_k
            )

            # Task 4: fact 提取（依赖 query，独立）
            future_fact = executor.submit(
                self._extract_fact_entities,
                query=query,
                top_k=linking_top_k
            )

            # 收集结果
            query_embedding = future_embedding.result()
            logger.info(f"  ✓ [并行] Query embedding 完成")

            semantic_entities = future_semantic.result()
            logger.info(f"  ✓ [并行] 语义扩展找到 {len(semantic_entities)} 个相似实体")

            emotion_entities = future_emotion.result()
            logger.info(f"  ✓ [并行] 情绪扩展找到 {len(emotion_entities)} 个实体")

            fact_entities = future_fact.result()
            logger.info(f"  ✓ [并行] Fact 扩展找到 {len(fact_entities)} 个实体")

        return semantic_entities, emotion_entities, fact_entities

    def _get_query_embedding_task(self, query: str):
        """获取 query embedding 的任务包装器"""
        self._hipporag_core.get_query_embeddings([query])
        return self._hipporag_core.query_to_embedding['triple'].get(query)

    def _semantic_expansion(
        self,
        query_entities: List[str],
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        语义扩展：在 HippoRAG entity_embedding_store 中找相似实体

        Args:
            query_entities: 查询实体列表
            top_k: 返回 top-k 相似实体

        Returns:
            {entity_text: similarity_score}
        """
        semantic_entities = defaultdict(float)

        # 获取所有实体节点
        entity_node_keys = list(self._hipporag_core.entity_node_keys)
        entity_embeddings = self._hipporag_core.entity_embeddings

        # 对每个 query entity 找相似实体
        # 重要：统一转换为小写，因为 HippoRAG 的 OpenIE 会将实体小写化
        for entity in query_entities:
            # 获取 entity embedding
            entity_id = compute_mdhash_id(content=entity.lower(), prefix="entity-")

            # 如果 entity 在 embedding store 中，找其相似实体
            if entity_id in self._hipporag_core.entity_embedding_store.hash_id_to_row:
                entity_emb = self._hipporag_core.entity_embedding_store.get_embeddings([entity_id])[0]

                # KNN 检索
                knn_results = retrieve_knn(
                    query_ids=[entity_id],
                    key_ids=entity_node_keys,
                    query_vecs=[entity_emb],
                    key_vecs=entity_embeddings,
                    k=top_k
                )

                if entity_id in knn_results:
                    similar_ids, scores = knn_results[entity_id]
                    for sim_id, score in zip(similar_ids, scores):
                        sim_entity = self._hipporag_core.entity_embedding_store.get_row(sim_id)['content']
                        semantic_entities[sim_entity] += score

        # 归一化
        if semantic_entities:
            max_score = max(semantic_entities.values())
            semantic_entities = {k: v / max_score for k, v in semantic_entities.items()}

        return semantic_entities

    def _emotion_expansion(
        self,
        query_particle,
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        情绪扩展：使用 Amygdala 找到情绪相似的粒子，提取标准 entity_id

        Args:
            query_particle: 查询粒子
            top_k: 返回 top-k 相似粒子

        Returns:
            {entity_text: emotion_score}
        """
        from poincare.retrieval import HyperAmyRetrieval

        emotion_entities = defaultdict(float)

        # 使用 Amygdala 检索
        retriever = HyperAmyRetrieval(
            storage=self.amygda.particle_storage,
            projector=self.amygda.particle_projector
        )

        search_results = retriever.search(
            query_entity=query_particle,
            top_k=top_k,
            cone_width=50
        )

        # 提取粒子的 entity 文本和标准 entity_id
        for result in search_results:
            # score 是双曲距离，越小越相似，转换为相似度分数
            similarity = 1.0 / (1.0 + result.score)
            entity_text = result.metadata.get("entity", "")
            standard_entity_id = result.metadata.get("entity_id", "")

            # 优先使用标准 entity_id（兼容 HippoRAG）
            # fallback 到 entity 文本
            if entity_text:
                # 使用 entity_text 作为键（因为 HippoRAG 的图谱使用文本内容）
                emotion_entities[entity_text] += similarity

                logger.debug(f"[GraphFusion._emotion_expansion] "
                           f"entity={entity_text}, "
                           f"standard_entity_id={standard_entity_id}, "
                           f"similarity={similarity:.4f}")

        # 归一化
        if emotion_entities:
            max_score = max(emotion_entities.values())
            emotion_entities = {k: v / max_score for k, v in emotion_entities.items()}

        logger.info(f"[GraphFusion._emotion_expansion] 找到 {len(emotion_entities)} 个情绪实体")

        return emotion_entities

    def _emotion_expansion_from_entities(
        self,
        query: str,
        query_entities: List[str],
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        从预抽取的实体进行情绪扩展

        使用统一抽取的实体，生成查询粒子进行情绪相似度搜索。

        Args:
            query: 查询文本
            query_entities: 统一抽取的实体列表
            top_k: 返回 top-k 相似粒子

        Returns:
            {entity_text: emotion_score}
        """
        from poincare.retrieval import HyperAmyRetrieval
        from particle.particle import Particle
        from particle.emotion_v2 import EmotionV2

        emotion_entities = defaultdict(float)

        if not query_entities:
            return emotion_entities

        try:
            # 使用 Amygdala 的 EmotionV2 为查询实体生成 emotion 向量
            # 注意：这里需要调用 LLM 生成情感描述和 embedding
            # 这是情绪扩展的必要步骤，无法省略
            emotion_v2 = EmotionV2(
                model_name=self.llm_model_name,
                embedding_model_name=self.embedding_model_name
            )

            # 为查询文本生成 emotion nodes
            emotion_nodes = emotion_v2.process(
                text=query,
                text_id=f"query_emotion_{int(time.time())}",
                entities=query_entities  # 使用预抽取的实体
            )

            if not emotion_nodes:
                logger.info(f"[GraphFusion._emotion_expansion_from_entities] 未生成 emotion nodes")
                return emotion_entities

            # 使用第一个 emotion node 作为查询粒子
            query_emotion = emotion_nodes[0]

            # 创建临时查询粒子（只包含 emotion 向量）
            class QueryParticle:
                def __init__(self, entity, emotion_vector):
                    self.entity = entity
                    self.emotion_vector = emotion_vector
                    self.entity_id = compute_mdhash_id(content=entity, prefix="entity-")

            query_particle = QueryParticle(
                entity=query_entities[0],  # 使用第一个实体
                emotion_vector=query_emotion.emotion_vector
            )

            # 使用 HyperAmyRetrieval 进行情绪相似度搜索
            retriever = HyperAmyRetrieval(
                storage=self.amygda.particle_storage,
                projector=self.amygda.particle_projector
            )

            search_results = retriever.search(
                query_entity=query_particle,
                top_k=top_k,
                cone_width=50
            )

            # 提取粒子的 entity 文本
            for result in search_results:
                similarity = 1.0 / (1.0 + result.score)
                entity_text = result.metadata.get("entity", "")

                if entity_text:
                    emotion_entities[entity_text] += similarity

            # 归一化
            if emotion_entities:
                max_score = max(emotion_entities.values())
                emotion_entities = {k: v / max_score for k, v in emotion_entities.items()}

            logger.info(f"[GraphFusion._emotion_expansion_from_entities] 找到 {len(emotion_entities)} 个情绪实体")

        except Exception as e:
            logger.warning(f"[GraphFusion._emotion_expansion_from_entities] 情绪扩展失败: {e}")
            import traceback
            traceback.print_exc()

        return emotion_entities

    def _extract_fact_entities(
        self,
        query: str,
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        从 HippoRAG facts 中提取实体

        Args:
            query: 查询文本
            top_k: 返回 top-k facts

        Returns:
            {entity_text: fact_score}
        """
        fact_entities = defaultdict(float)

        # 计算 fact scores
        query_fact_scores = self._hipporag_core.get_fact_scores(query)

        if len(query_fact_scores) == 0:
            return fact_entities

        # 确保 scores 是 1D numpy array
        query_fact_scores = np.asarray(query_fact_scores).flatten()

        # 获取 top-k facts 的索引
        link_top_k = min(top_k, len(query_fact_scores))
        # 使用 argsort 获取排序后的索引
        sorted_indices = np.argsort(query_fact_scores)
        # 获取 top-k 个最高分数的索引（倒序）
        top_k_indices = sorted_indices[-link_top_k:][::-1]

        # 提取实体
        for idx in top_k_indices:
            # 确保 idx 是标量
            if hasattr(idx, 'item'):
                idx_int = int(idx.item())
            else:
                idx_int = int(idx)

            # 边界检查
            if idx_int < 0 or idx_int >= len(self._hipporag_core.fact_node_keys):
                continue

            fact_id = self._hipporag_core.fact_node_keys[idx_int]
            fact_row = self._hipporag_core.fact_embedding_store.get_row(fact_id)
            fact = eval(fact_row['content'])  # (subject, predicate, object)

            fact_score = float(query_fact_scores[idx_int])

            # 提取 subject 和 object
            for entity in [fact[0], fact[2]]:
                entity_lower = entity.lower()
                fact_entities[entity_lower] += fact_score

        # 归一化
        if fact_entities:
            max_score = max(fact_entities.values())
            fact_entities = {k: v / max_score for k, v in fact_entities.items()}

        return fact_entities

    def _merge_entity_weights(
        self,
        query_entities: List[str],
        semantic_entities: Dict[str, float],
        emotion_entities: Dict[str, float],
        fact_entities: Dict[str, float],
        emotion_weight: float,
        semantic_weight: float,
        fact_weight: float
    ) -> np.ndarray:
        """
        融合实体权重到 HippoRAG 图谱节点

        Returns:
            entity_weights: np.ndarray, shape=(num_nodes,)
        """
        # 合并所有实体
        all_entities = set(query_entities)
        all_entities.update(semantic_entities.keys())
        all_entities.update(emotion_entities.keys())
        all_entities.update(fact_entities.keys())

        logger.info(f"    - Query 实体: {len(query_entities)}")
        logger.info(f"    - 语义实体: {len(semantic_entities)}")
        logger.info(f"    - 情绪实体: {len(emotion_entities)}")
        logger.info(f"    - Fact 实体: {len(fact_entities)}")
        logger.info(f"    - 合并后唯一实体: {len(all_entities)}")

        # 初始化权重数组
        num_nodes = len(self._hipporag_core.graph.vs['name'])
        entity_weights = np.zeros(num_nodes)

        # 映射到 HippoRAG 节点并分配权重
        node_name_to_idx = self._hipporag_core.node_name_to_vertex_idx

        for entity in all_entities:
            # 重要：统一转换为小写，因为 HippoRAG 的 OpenIE 会将实体小写化
            entity_key = compute_mdhash_id(content=entity.lower(), prefix="entity-")

            if entity_key not in node_name_to_idx:
                continue

            node_idx = node_name_to_idx[entity_key]

            # 计算融合权重
            weight = 0.0

            # Query 实体：最高权重
            if entity in query_entities:
                weight += 1.0

            # 语义扩展
            if entity in semantic_entities:
                weight += semantic_weight * semantic_entities[entity]

            # 情绪扩展
            if entity in emotion_entities:
                weight += emotion_weight * emotion_entities[entity]

            # Fact 扩展
            if entity in fact_entities:
                weight += fact_weight * fact_entities[entity]

            # 考虑实体出现频率（类似 HippoRAG 的处理）
            chunk_ids = self._hipporag_core.ent_node_to_chunk_ids.get(entity_key, set())
            if len(chunk_ids) > 0:
                weight /= len(chunk_ids)

            entity_weights[node_idx] += weight

        return entity_weights

    def _run_ppr_with_entity_weights(
        self,
        entity_weights: np.ndarray,
        passage_node_weight: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用融合的实体权重运行 PPR

        Args:
            entity_weights: 实体节点权重
            passage_node_weight: passage 节点权重

        Returns:
            (sorted_doc_ids, ppr_scores)
        """
        # 获取 DPR passage scores
        query_text = ""  # 这里需要传入 query，但为了简化，先不使用
        # 注意：这里应该有 query，但由于是内部方法，我们从外部获取

        # 为了简化，这里只使用 entity_weights
        # 在实际使用中，应该加入 DPR scores

        # 运行 PPR
        sorted_doc_ids, ppr_scores = self._hipporag_core.run_ppr(
            reset_prob=entity_weights,
            damping=self._hipporag_core.global_config.damping
        )

        return sorted_doc_ids, ppr_scores

    def _format_results(
        self,
        sorted_doc_ids: np.ndarray,
        ppr_scores: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        格式化检索结果

        Args:
            sorted_doc_ids: 排序后的文档 ID 列表
            ppr_scores: PPR 分数
            top_k: 返回 top-k

        Returns:
            格式化的结果列表
        """
        results = []

        for rank, doc_id in enumerate(sorted_doc_ids[:top_k]):
            passage_node_key = self._hipporag_core.passage_node_keys[doc_id]
            passage_text = self._hipporag_core.chunk_embedding_store.get_row(passage_node_key)['content']

            results.append({
                'rank': rank + 1,
                'text': passage_text,
                'score': float(ppr_scores[doc_id]),
                'conversation_id': passage_node_key
            })

        return results

    def _fallback_to_hipporag(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        回退到 HippoRAG 直接检索（当 query 未生成粒子时）

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            HippoRAG 检索结果
        """
        logger.warning(f"[GraphFusionRetriever] 回退到 HippoRAG 直接检索")
        hipporag_results = self.hipporag.retrieve(query=query, top_k=top_k)

        # 转换格式
        results = []
        for rank, result in enumerate(hipporag_results):
            results.append({
                'rank': rank + 1,
                'text': result['text'],
                'score': result['score'],
                'conversation_id': result.get('doc_id', '')
            })

        return results

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


def create_graph_fusion_retriever(
    amygdala_save_dir: str = "./graph_fusion_amygdala_db",
    hipporag_save_dir: str = "./graph_fusion_hipporag_db",
    llm_model_name: str = "DeepSeek-V3.2",
    embedding_model_name: str = None
) -> GraphFusionRetriever:
    """
    便捷函数：创建图谱融合检索器

    Args:
        amygdala_save_dir: Amygdala 数据目录
        hipporag_save_dir: HippoRAG 数据目录
        llm_model_name: LLM 模型名称
        embedding_model_name: 嵌入模型名称

    Returns:
        GraphFusionRetriever 实例
    """
    return GraphFusionRetriever(
        amygdala_save_dir=amygdala_save_dir,
        hipporag_save_dir=hipporag_save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name or DEFAULT_EMBEDDING_MODEL
    )
