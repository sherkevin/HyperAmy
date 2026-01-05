#!/usr/bin/env python3
"""
融合检索对比测试 - 简化版

对比 GraphFusion vs HippoRAG vs Amygdala 的检索效果
"""

import logging
import sys
import time
from pathlib import Path

# 设置日志文件
log_file = Path("./log/test_fusion_comparison_simple.log")
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
import os
os.environ["OPENAI_API_KEY"] = API_KEY

from workflow.graph_fusion_retrieval import GraphFusionRetriever
from workflow.amygdala import Amygdala
from workflow.hipporag_wrapper import HippoRAGWrapper
from poincare.retrieval import HyperAmyRetrieval

logger.info("=" * 100)
logger.info("融合检索对比测试（简化版）：GraphFusion vs HippoRAG vs Amygdala")
logger.info("=" * 100)

# 测试数据
chunks = [
    '"I have an excellent appetite," said Albert. "I hope, my dear Count, you have the same." '
    '"I?" said Monte Cristo. "I never eat, or rather, I eat so little that it is not worth '
    'talking about. I have my own peculiar habits."',

    'The Count took from his pocket a small case made of hollowed emerald, took out a small '
    'greenish pill, and swallowed it. "This is my food," he said to the guests. "With this, '
    'I feel neither hunger nor fatigue. It is a secret I learned in the East."',

    '"Will you not take anything?" asked Mercedes. "A peach? Some grapes?" '
    '"I thank you, Madame," replied Monte Cristo with a bow, "but I never eat between meals. '
    'It is a rule I have imposed upon myself to maintain my health."',

    '"In the countries of the East, where I have lived," said Monte Cristo to Franz, '
    '"people who eat and drink together are bound by a sacred tie. They become brothers. '
    'Therefore, I never eat or drink in the house of a man whom I wish to kill. '
    'If I shared their bread, I would be forbidden by honor to take my revenge."',

    'Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate. '
    '"You refuse?" she whispered, her voice full of tears. "Is it because you are our enemy? '
    'To refuse to break bread... means you bring death to this house." '
    'She realized then that the man standing before her was not just a visitor, but an avenger '
    'who remembered the past.'
]

chunk_types = {
    0: "早餐场景",
    1: "药丸场景",
    2: "拒绝葡萄干",
    3: "东方哲学（核心答案）",
    4: "情感对峙"
}

query = "Why did the Count strictly refuse the muscatel grapes and any refreshment offered by Madame de Morcerf (Mercedes) during his visit to her house?"

# 初始化融合检索器（包含两个系统）
logger.info("\n" + "=" * 100)
logger.info("【初始化】图谱融合检索器")
logger.info("=" * 100)

fusion = GraphFusionRetriever(
    amygdala_save_dir="./test_simple_fusion_amygdala_db",
    hipporag_save_dir="./test_simple_fusion_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    auto_link_particles=False
)

# 添加数据
logger.info("\n【添加数据】")
add_start = time.time()
result = fusion.add(chunks)
add_time = time.time() - add_start
logger.info(f"✓ 数据添加完成 ({add_time:.2f}s)")
logger.info(f"  - Amygdala 粒子数: {result['amygdala_count']}")
logger.info(f"  - HippoRAG chunks: {result['hipporag_count']}")

# 获取内部系统引用
amygdala = fusion.amygda
hipporag = fusion.hipporag

# 辅助函数
def get_chunk_type(chunk_text):
    for i, chunk in enumerate(chunks):
        if chunk_text == chunk:
            return chunk_types[i]
    return "Unknown"

def get_rank(results, chunk_idx):
    for result in results:
        for i, chunk in enumerate(chunks):
            if result['text'] == chunk and i == chunk_idx:
                return result['rank']
    return None

# ========== 测试 1: 图谱融合检索 ==========
logger.info("\n" + "=" * 100)
logger.info("【测试 1】图谱融合检索")
logger.info("=" * 100)
logger.info(f"Query: {query[:100]}...")

start_time = time.time()
fusion_results = fusion.retrieve(
    query=query,
    top_k=5,
    emotion_weight=0.3,
    semantic_weight=0.5,
    fact_weight=0.2
)
fusion_time = time.time() - start_time

logger.info(f"\n【融合检索结果】（耗时: {fusion_time:.2f}s）")
for result in fusion_results:
    chunk_type = get_chunk_type(result['text'])
    logger.info(f"  Rank {result['rank']}: {chunk_type} (分数: {result['score']:.4f})")

# ========== 测试 2: HippoRAG 单独检索 ==========
logger.info("\n" + "=" * 100)
logger.info("【测试 2】HippoRAG 单独检索")
logger.info("=" * 100)

start_time = time.time()
hipporag_results_raw = hipporag.retrieve(query=query, top_k=5)
hipporag_time = time.time() - start_time

hipporag_results = []
for rank, result in enumerate(hipporag_results_raw):
    hipporag_results.append({
        'rank': rank + 1,
        'text': result['text'],
        'score': result['score']
    })

logger.info(f"\n【HippoRAG 结果】（耗时: {hipporag_time:.2f}s）")
for result in hipporag_results:
    chunk_type = get_chunk_type(result['text'])
    logger.info(f"  Rank {result['rank']}: {chunk_type} (分数: {result['score']:.4f})")

# ========== 测试 3: Amygdala 单独检索 ==========
logger.info("\n" + "=" * 100)
logger.info("【测试 3】Amygdala 单独检索")
logger.info("=" * 100)

start_time = time.time()
query_particles = amygdala.particle.process(
    text=query,
    text_id=f"query_{int(time.time())}"
)

amygdala_results = []
if query_particles:
    retriever = HyperAmyRetrieval(
        storage=amygdala.particle_storage,
        projector=amygdala.particle_projector
    )

    search_results = retriever.search(
        query_entity=query_particles[0],
        top_k=5,
        cone_width=50
    )

    # 获取所有 conversation_id
    conversation_ids = [r.metadata.get("conversation_id", "") for r in search_results]

    # 批量获取对话文本
    if conversation_ids:
        conversations = amygdala.conversation_store.get_strings_by_ids(conversation_ids)

        # 创建 conversation_id 到文本的映射
        conv_to_text = {}
        for conv in conversations:
            conv_id = conv.get('id', '')
            text = conv.get('text', '')
            if conv_id and text:
                conv_to_text[conv_id] = text

        for rank, result in enumerate(search_results):
            conv_id = result.metadata.get("conversation_id", "")
            chunk_text = conv_to_text.get(conv_id, "")

            if chunk_text:
                amygdala_results.append({
                    'rank': rank + 1,
                    'text': chunk_text,
                    'score': result.score
                })

amygdala_time = time.time() - start_time

logger.info(f"\n【Amygdala 结果】（耗时: {amygdala_time:.2f}s）")
if amygdala_results:
    for result in amygdala_results:
        chunk_type = get_chunk_type(result['text'])
        logger.info(f"  Rank {result['rank']}: {chunk_type} (距离: {result['score']:.4f})")
else:
    logger.info("  未检索到结果")

# ========== 对比分析 ==========
logger.info("\n" + "=" * 100)
logger.info("【对比分析】")
logger.info("=" * 100)

# 关键 chunks
key_chunks = {
    "情感对峙": 4,
    "东方哲学": 3,
    "拒绝葡萄干": 2
}

all_modes = [
    ("图谱融合", fusion_results),
    ("HippoRAG", hipporag_results),
    ("Amygdala", amygdala_results)
]

# 排名对比
logger.info("\n1. 排名对比:")
logger.info(f"{'检索方式':<15} {'情感对峙':<12} {'东方哲学':<12} {'拒绝葡萄干':<12} {'平均排名':<12}")
logger.info("-" * 70)

for mode_name, results in all_modes:
    ranks = []
    for chunk_name, chunk_idx in key_chunks.items():
        rank = get_rank(results, chunk_idx)
        rank_str = str(rank) if rank else "-"
        if rank:
            ranks.append(rank)
        logger.info(f"{mode_name:<15} {rank_str:<12}", end="")
    logger.info(f" {sum(ranks)/len(ranks):<12.2f}" if ranks else f" {'-':<12}")

# Top-3 命中率
logger.info("\n2. Top-3 命中率（关键 chunks）:")
key_chunk_indices = [4, 3, 2]

for mode_name, results in all_modes:
    top_3_hits = sum(1 for chunk_idx in key_chunk_indices
                     if (rank := get_rank(results, chunk_idx)) and rank <= 3)
    hit_rate = top_3_hits / len(key_chunk_indices) * 100
    logger.info(f"  {mode_name:<15}: {top_3_hits}/{len(key_chunk_indices)} 命中 (命中率: {hit_rate:.1f}%)")

# 性能对比
logger.info("\n3. 检索性能对比:")
times = [
    ("图谱融合", fusion_time),
    ("HippoRAG", hipporag_time),
    ("Amygdala", amygdala_time)
]

min_time = min(t[1] for t in times)
for mode_name, time_cost in times:
    relative = time_cost / min_time
    logger.info(f"  {mode_name:<15}: {time_cost:.2f}s ({relative:.2f}x)")

# Top-1 准确率（情感对峙）
logger.info("\n4. Top-1 准确率（情感对峙）:")
for mode_name, results in all_modes:
    rank = get_rank(results, 4)
    if rank == 1:
        logger.info(f"  {mode_name:<15}: ✓ Rank 1 (最佳)")
    elif rank:
        logger.info(f"  {mode_name:<15}: Rank {rank}")
    else:
        logger.info(f"  {mode_name:<15}: ✗ 未检索到")

# ========== 总结 ==========
logger.info("\n" + "=" * 100)
logger.info("【总结】")
logger.info("=" * 100)

logger.info("\n✅ 测试完成！")
logger.info(f"\n日志已保存到: {log_file}")
logger.info("\n主要发现:")
logger.info("  • 融合检索结合了 HippoRAG 的语义理解和 Amygdala 的情绪感知")
logger.info("  • 可通过调整 emotion_weight、semantic_weight、fact_weight 优化效果")
logger.info("  • 建议根据具体场景选择合适的检索方式")
