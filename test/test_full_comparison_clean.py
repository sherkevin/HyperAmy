#!/usr/bin/env python3
"""
完整测试脚本：创建新数据 + 运行三种检索方案对比

1. 创建新的测试数据库（Monte Cristo场景）
2. 运行 GraphFusion, HippoRAG, Amygdala 对比测试
3. 生成完整的测试报告
"""
import os
import sys
import time
import shutil
from pathlib import Path

# 设置环境变量
from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
os.environ["OPENAI_API_KEY"] = API_KEY

import logging
log_file = Path("./log/test_full_comparison_clean.log")
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

from workflow.graph_fusion_retrieval import GraphFusionRetriever
from workflow.amygdala import Amygdala
from workflow.hipporag_wrapper import HippoRAGWrapper
from poincare.retrieval import HyperAmyRetrieval

logger.info("=" * 120)
logger.info("完整测试：创建新数据 + 三种检索方案对比")
logger.info("=" * 120)

# 测试数据（Monte Cristo场景）
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

# ========== Step 1: 清理旧数据 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 1】清理旧数据")
logger.info("=" * 120)

test_databases = [
    "./test_full_amygdala_db",
    "./test_full_hipporag_db",
    "./hyperamy_db_full_particles"
]

for db_dir in test_databases:
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        logger.info(f"✓ 删除 {db_dir}")

logger.info("✓ 旧数据清理完成")

# ========== Step 2: 创建 Amygdala 数据库 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 2】创建 Amygdala 数据库")
logger.info("=" * 120)

amygdala = Amygdala(
    save_dir="./test_full_amygdala_db",
    particle_collection_name="full_test_particles",
    conversation_namespace="full_test",
    auto_link_particles=False
)
logger.info("✓ Amygdala 初始化完成")

# 添加数据
logger.info(f"\n添加 {len(chunks)} 个对话...")
for i, chunk in enumerate(chunks):
    result = amygdala.add(chunk)
    logger.info(f"  Chunk {i+1}: 生成 {result['particle_count']} 个粒子")

logger.info(f"✓ Amygdala 数据库创建完成（{len(chunks)} 个对话）")

# ========== Step 3: 创建 HippoRAG 数据库 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 3】创建 HippoRAG 数据库")
logger.info("=" * 120)

hipporag = HippoRAGWrapper(
    save_dir="./test_full_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    llm_base_url=BASE_URL,
    embedding_model_name=f"VLLM/{DEFAULT_EMBEDDING_MODEL}",
    embedding_base_url=API_URL_EMBEDDINGS
)
logger.info("✓ HippoRAG 初始化完成")

# 添加数据
logger.info(f"\n添加 {len(chunks)} 个对话...")
hipporag.add(chunks)
logger.info(f"✓ HippoRAG 数据库创建完成（{len(chunks)} 个对话）")

# ========== Step 4: 创建 GraphFusion ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 4】创建 GraphFusion 检索器")
logger.info("=" * 120)

fusion = GraphFusionRetriever(
    amygdala_save_dir="./test_full_amygdala_db",
    hipporag_save_dir="./test_full_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    auto_link_particles=False
)
logger.info("✓ GraphFusion 初始化完成")

# ========== Step 5: 测试检索 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 5】测试三种检索方案")
logger.info("=" * 120)
logger.info(f"Query: {query}")

# 测试 1: GraphFusion
logger.info("\n" + "-" * 120)
logger.info("【测试 1】GraphFusion 检索")
logger.info("-" * 120)

start_time = time.time()
fusion_results = fusion.retrieve(
    query=query,
    top_k=5,
    emotion_weight=0.3,
    semantic_weight=0.5,
    fact_weight=0.2
)
fusion_time = time.time() - start_time

logger.info(f"✓ GraphFusion 检索完成（耗时: {fusion_time:.2f}s）")
for result in fusion_results:
    chunk_type = chunk_types.get(chunks.index(result['text']), "Unknown")
    logger.info(f"  Rank {result['rank']}: {chunk_type} (PPR分数: {result['score']:.4f})")

# 测试 2: HippoRAG
logger.info("\n" + "-" * 120)
logger.info("【测试 2】HippoRAG 检索")
logger.info("-" * 120)

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

logger.info(f"✓ HippoRAG 检索完成（耗时: {hipporag_time:.2f}s）")
for result in hipporag_results:
    chunk_type = chunk_types.get(chunks.index(result['text']), "Unknown")
    logger.info(f"  Rank {result['rank']}: {chunk_type} (分数: {result['score']:.4f})")

# 测试 3: Amygdala
logger.info("\n" + "-" * 120)
logger.info("【测试 3】Amygdala 检索")
logger.info("-" * 120)

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
        cone_width=20  # 使用较小的值
    )

    # 获取所有 conversation_id
    conversation_ids = [r.metadata.get("conversation_id", "") for r in search_results]

    # 创建 conversation_id 到文本的映射
    conv_to_text = {}
    for conv_id in conversation_ids:
        if conv_id:
            text = amygdala.conversation_store.get_text(conv_id)
            if text:
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

logger.info(f"✓ Amygdala 检索完成（耗时: {amygdala_time:.2f}s）")
if amygdala_results:
    for result in amygdala_results:
        chunk_type = chunk_types.get(chunks.index(result['text']), "Unknown")
        logger.info(f"  Rank {result['rank']}: {chunk_type} (双曲距离: {result['score']:.4f})")
else:
    logger.info("  ⚠️ 未检索到结果")

# ========== Step 6: 对比分析 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 6】对比分析")
logger.info("=" * 120)

# 辅助函数
def get_chunk_type(chunk_text):
    for i, chunk in enumerate(chunks):
        if chunk_text == chunk:
            return chunk_types[i]
    return "Unknown"

def get_rank(results, chunk_idx):
    for result in results:
        if result.get('text') == chunks[chunk_idx]:
            return result['rank']
    return None

# 排名对比
all_modes = [
    ("GraphFusion", fusion_results),
    ("HippoRAG", hipporag_results),
    ("Amygdala", amygdala_results)
]

logger.info("\n排名对比:")
logger.info(f"{'检索方式':<15} {'情感对峙':<12} {'东方哲学':<12} {'拒绝葡萄干':<12} {'药丸场景':<12} {'早餐场景':<12}")
logger.info("-" * 100)

for mode_name, results in all_modes:
    row_str = f"{mode_name:<15} "
    for chunk_idx in range(5):
        rank = get_rank(results, chunk_idx)
        rank_str = str(rank) if rank else "-"
        row_str += f"{rank_str:<12}"
    logger.info(row_str)

# 性能对比
logger.info("\n性能对比:")
logger.info(f"{'检索方式':<15} {'时间(s)':<15} {'相对倍数':<15}")
logger.info("-" * 50)

times = [
    ("GraphFusion", fusion_time),
    ("HippoRAG", hipporag_time),
    ("Amygdala", amygdala_time)
]

min_time = min(t[1] for t in times)
for mode_name, time_cost in times:
    relative = time_cost / min_time
    logger.info(f"{mode_name:<15} {time_cost:<15.2f} {relative:<15.2f}x")

# ========== 完成 ==========
logger.info("\n" + "=" * 120)
logger.info("✓ 测试完成！")
logger.info("=" * 120)
logger.info(f"\n日志已保存到: {log_file}")
