#!/usr/bin/env python3
"""
融合检索对比测试 - GraphFusion vs HippoRAG vs Amygdala

测试场景：
1. 图谱融合检索（HippoRAG + Amygdala 实体级融合）
2. 单独使用 HippoRAG
3. 单独使用 Amygdala
4. 详细效果对比分析

使用 Monte Cristo 数据集
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# 设置日志文件
log_file = Path("./log/test_graph_fusion_comparison.log")
log_file.parent.mkdir(exist_ok=True)

# 配置日志同时输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 设置环境变量
from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
import os
os.environ["OPENAI_API_KEY"] = API_KEY

from workflow.graph_fusion_retrieval import GraphFusionRetriever
from workflow.amygdala import Amygdala
from workflow.hipporag_wrapper import HippoRAGWrapper
from poincare.retrieval import HyperAmyRetrieval

logger.info("=" * 100)
logger.info("融合检索对比测试：GraphFusion vs HippoRAG vs Amygdala")
logger.info("=" * 100)

# 测试数据（Monte Cristo 场景）
chunks = [
    # Chunk 1 - 早餐场景
    '"I have an excellent appetite," said Albert. "I hope, my dear Count, you have the same." '
    '"I?" said Monte Cristo. "I never eat, or rather, I eat so little that it is not worth '
    'talking about. I have my own peculiar habits."',

    # Chunk 2 - 药丸场景
    'The Count took from his pocket a small case made of hollowed emerald, took out a small '
    'greenish pill, and swallowed it. "This is my food," he said to the guests. "With this, '
    'I feel neither hunger nor fatigue. It is a secret I learned in the East."',

    # Chunk 3 - 花园里的拒绝场景
    '"Will you not take anything?" asked Mercedes. "A peach? Some grapes?" '
    '"I thank you, Madame," replied Monte Cristo with a bow, "but I never eat between meals. '
    'It is a rule I have imposed upon myself to maintain my health."',

    # Chunk 4 - 东方哲学（核心答案）
    '"In the countries of the East, where I have lived," said Monte Cristo to Franz, '
    '"people who eat and drink together are bound by a sacred tie. They become brothers. '
    'Therefore, I never eat or drink in the house of a man whom I wish to kill. '
    'If I shared their bread, I would be forbidden by honor to take my revenge."',

    # Chunk 5 - 情感对峙
    'Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate. '
    '"You refuse?" she whispered, her voice full of tears. "Is it because you are our enemy? '
    'To refuse to break bread... means you bring death to this house." '
    'She realized then that the man standing before her was not just a visitor, but an avenger '
    'who remembered the past.'
]

# 定义 chunk 类型
chunk_types = {
    0: "早餐场景",
    1: "药丸场景",
    2: "拒绝葡萄干",
    3: "东方哲学（核心答案）",
    4: "情感对峙"
}

query = "Why did the Count strictly refuse the muscatel grapes and any refreshment offered by Madame de Morcerf (Mercedes) during his visit to her house?"

# ========== 初始化三个系统 ==========
logger.info("\n" + "=" * 100)
logger.info("【初始化】三个检索系统")
logger.info("=" * 100)

# 1. 图谱融合检索器
logger.info("\n[1/3] 初始化图谱融合检索器...")
fusion = GraphFusionRetriever(
    amygdala_save_dir="./test_comparison_fusion_amygdala_db",
    hipporag_save_dir="./test_comparison_fusion_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    auto_link_particles=False
)
logger.info("✓ 图谱融合检索器初始化完成")

# 2. 单独的 HippoRAG
logger.info("\n[2/3] 初始化 HippoRAG...")
hipporag = HippoRAGWrapper(
    save_dir="./test_comparison_hipporag_only_db",
    llm_model_name="DeepSeek-V3.2",
    llm_base_url=BASE_URL,
    embedding_model_name=f"VLLM/{DEFAULT_EMBEDDING_MODEL}",
    embedding_base_url=API_URL_EMBEDDINGS
)
logger.info("✓ HippoRAG 初始化完成")

# 3. 单独的 Amygdala
logger.info("\n[3/3] 初始化 Amygdala...")
amygdala = Amygdala(
    save_dir="./test_comparison_amygdala_only_db",
    particle_collection_name="comparison_particles",
    conversation_namespace="comparison",
    auto_link_particles=False
)
logger.info("✓ Amygdala 初始化完成")

# ========== 添加数据到三个系统 ==========
logger.info("\n" + "=" * 100)
logger.info("【添加数据】")
logger.info("=" * 100)

add_start = time.time()

# 添加到融合检索器
logger.info("\n[1/3] 添加数据到融合检索器...")
fusion_result = fusion.add(chunks)
logger.info(f"✓ 融合检索器: Amygdala {fusion_result['amygdala_count']} 粒子, HippoRAG {fusion_result['hipporag_count']} chunks")

# 添加到 HippoRAG
logger.info("\n[2/3] 添加数据到 HippoRAG...")
hipporag_result = hipporag.add(chunks)
logger.info(f"✓ HippoRAG: {hipporag_result['total_indexed']} chunks")

# 添加到 Amygdala
logger.info("\n[3/3] 添加数据到 Amygdala...")
amygdala_count = 0
for chunk in chunks:
    result = amygdala.add(chunk)
    amygdala_count += result['particle_count']
logger.info(f"✓ Amygdala: {amygdala_count} 粒子")

add_time = time.time() - add_start
logger.info(f"\n✓ 所有数据添加完成 (总耗时: {add_time:.2f}s)")

# ========== 辅助函数 ==========
def get_chunk_type(chunk_text):
    """根据 chunk 文本返回类型"""
    for i, chunk in enumerate(chunks):
        if chunk_text == chunk:
            return chunk_types[i]
    return "Unknown"

def get_rank(results, chunk_idx):
    """获取指定 chunk 的排名"""
    for result in results:
        for i, chunk in enumerate(chunks):
            if result['text'] == chunk and i == chunk_idx:
                return result['rank']
    return None

def format_results(results, name):
    """格式化检索结果"""
    logger.info(f"\n【{name}检索结果】")
    for result in results:
        chunk_type = get_chunk_type(result['text'])
        score = result.get('score', 0)
        logger.info(f"  Rank {result['rank']}: {chunk_type}")
        logger.info(f"    - 分数: {score:.4f}")

# ========== 测试 1: 图谱融合检索 ==========
logger.info("\n" + "=" * 100)
logger.info("【测试 1】图谱融合检索（GraphFusion）")
logger.info("=" * 100)
logger.info(f"\nQuery: {query}")
logger.info(f"权重配置: emotion=0.3, semantic=0.5, fact=0.2\n")

start_time = time.time()
fusion_results = fusion.retrieve(
    query=query,
    top_k=5,
    emotion_weight=0.3,
    semantic_weight=0.5,
    fact_weight=0.2
)
fusion_time = time.time() - start_time

format_results(fusion_results, "融合检索")
logger.info(f"\n⏱️ 耗时: {fusion_time:.2f}s")

# ========== 测试 2: HippoRAG 单独检索 ==========
logger.info("\n" + "=" * 100)
logger.info("【测试 2】HippoRAG 单独检索")
logger.info("=" * 100)
logger.info(f"\nQuery: {query}\n")

start_time = time.time()
hipporag_results_raw = hipporag.retrieve(query=query, top_k=5)
hipporag_time = time.time() - start_time

# 转换格式
hipporag_results = []
for rank, result in enumerate(hipporag_results_raw):
    hipporag_results.append({
        'rank': rank + 1,
        'text': result['text'],
        'score': result['score']
    })

format_results(hipporag_results, "HippoRAG")
logger.info(f"\n⏱️ 耗时: {hipporag_time:.2f}s")

# ========== 测试 3: Amygdala 单独检索 ==========
logger.info("\n" + "=" * 100)
logger.info("【测试 3】Amygdala 单独检索")
logger.info("=" * 100)
logger.info(f"\nQuery: {query}\n")

start_time = time.time()

# 处理 query 生成粒子
query_particles = amygdala.particle.process(
    text=query,
    text_id=f"query_{int(time.time())}"
)

if not query_particles:
    logger.warning("Query 未生成任何粒子")
    amygdala_results = []
else:
    # 使用 Amygdala 检索
    retriever = HyperAmyRetrieval(
        storage=amygdala.particle_storage,
        projector=amygdala.particle_projector
    )

    search_results = retriever.search(
        query_entity=query_particles[0],
        top_k=5,
        cone_width=50
    )

    # 转换格式
    amygdala_results = []
    for rank, result in enumerate(search_results):
        # 获取对应的 chunk 文本
        conversation_id = result.metadata.get("conversation_id", "")
        chunk_text = result.metadata.get("chunk_text", "")

        if chunk_text:
            amygdala_results.append({
                'rank': rank + 1,
                'text': chunk_text,
                'score': result.score  # 双曲距离，越小越相似
            })

amygdala_time = time.time() - start_time

format_results(amygdala_results, "Amygdala")
logger.info(f"\n⏱️ 耗时: {amygdala_time:.2f}s")

# ========== 详细对比分析 ==========
logger.info("\n" + "=" * 100)
logger.info("【详细对比分析】")
logger.info("=" * 100)

# 定义关键 chunks
key_chunks = {
    "情感对峙": 4,
    "东方哲学（核心答案）": 3,
    "拒绝葡萄干": 2,
    "药丸场景": 1,
    "早餐场景": 0
}

# 1. 排名对比
logger.info("\n" + "=" * 80)
logger.info("1. 排名对比")
logger.info("=" * 80)

logger.info(f"\n{'检索方式':<20} {'情感对峙':<12} {'东方哲学':<12} {'拒绝葡萄干':<12} {'药丸场景':<12} {'早餐场景':<12}")
logger.info("-" * 90)

all_modes = [
    ("图谱融合", fusion_results),
    ("HippoRAG", hipporag_results),
    ("Amygdala", amygdala_results)
]

for mode_name, results in all_modes:
    ranks = []
    for chunk_name, chunk_idx in key_chunks.items():
        rank = get_rank(results, chunk_idx)
        rank_str = str(rank) if rank else "-"
        ranks.append(rank_str)
        logger.info(f"{mode_name:<20} {rank_str:<12}", end="")
    logger.info()

# 计算平均排名
logger.info("\n平均排名（越低越好）:")
logger.info(f"{'检索方式':<20} {'平均排名':<15} {'标准差':<15}")
logger.info("-" * 60)

for mode_name, results in all_modes:
    ranks_list = []
    for chunk_idx in key_chunks.values():
        rank = get_rank(results, chunk_idx)
        if rank:
            ranks_list.append(rank)

    if ranks_list:
        avg_rank = sum(ranks_list) / len(ranks_list)
        variance = sum((r - avg_rank) ** 2 for r in ranks_list) / len(ranks_list)
        std_dev = variance ** 0.5
        logger.info(f"{mode_name:<20} {avg_rank:<15.2f} {std_dev:<15.2f}")
    else:
        logger.info(f"{mode_name:<20} {'-':<15} {'-':<15}")

# 2. Top-3 命中率
logger.info("\n" + "=" * 80)
logger.info("2. Top-3 命中率（关键 chunks）")
logger.info("=" * 80)

# 关键 chunks: 情感对峙、东方哲学、拒绝葡萄干
key_chunk_indices = [4, 3, 2]

for mode_name, results in all_modes:
    top_3_hits = 0
    for chunk_idx in key_chunk_indices:
        rank = get_rank(results, chunk_idx)
        if rank and rank <= 3:
            top_3_hits += 1

    hit_rate = top_3_hits / len(key_chunk_indices) * 100
    logger.info(f"{mode_name:<20} {top_3_hits}/{len(key_chunk_indices)} 命中 (命中率: {hit_rate:.1f}%)")

# 3. Top-1 准确率
logger.info("\n" + "=" * 80)
logger.info("3. Top-1 准确率（关键 chunks）")
logger.info("=" * 80)

# 最关键: 情感对峙（Rank 1）
target_chunk_idx = 4  # 情感对峙

for mode_name, results in all_modes:
    rank = get_rank(results, target_chunk_idx)
    if rank == 1:
        logger.info(f"{mode_name:<20} ✓ Rank 1 (最佳)")
    elif rank:
        logger.info(f"{mode_name:<20} Rank {rank} (偏差 {rank - 1})")
    else:
        logger.info(f"{mode_name:<20} ✗ 未检索到")

# 4. 检索时间对比
logger.info("\n" + "=" * 80)
logger.info("4. 检索性能对比")
logger.info("=" * 80)

times = [
    ("图谱融合", fusion_time),
    ("HippoRAG", hipporag_time),
    ("Amygdala", amygdala_time)
]

logger.info(f"\n{'检索方式':<20} {'时间(s)':<15} {'相对倍数':<15}")
logger.info("-" * 60)

min_time = min(t[1] for t in times)
for mode_name, time_cost in times:
    relative = time_cost / min_time
    logger.info(f"{mode_name:<20} {time_cost:<15.2f} {relative:<15.2f}x")

# ========== 总结分析 ==========
logger.info("\n" + "=" * 100)
logger.info("【总结分析】")
logger.info("=" * 100)

logger.info("\n📊 检索效果总结:")

# 计算各项指标
metrics = {}

for mode_name, results in all_modes:
    # 平均排名
    ranks_list = []
    for chunk_idx in key_chunks.values():
        rank = get_rank(results, chunk_idx)
        if rank:
            ranks_list.append(rank)

    if ranks_list:
        avg_rank = sum(ranks_list) / len(ranks_list)

        # Top-3 命中率
        top_3_hits = sum(1 for chunk_idx in key_chunk_indices
                        if (rank := get_rank(results, chunk_idx)) and rank <= 3)
        hit_rate = top_3_hits / len(key_chunk_indices) * 100

        # Top-1 准确率
        top_1_acc = 1.0 if get_rank(results, 4) == 1 else 0.0

        metrics[mode_name] = {
            'avg_rank': avg_rank,
            'top3_hit_rate': hit_rate,
            'top1_accuracy': top_1_acc
        }

logger.info(f"\n{'检索方式':<15} {'平均排名':<12} {'Top-3命中率':<15} {'Top-1准确率':<12}")
logger.info("-" * 65)

for mode_name, metric in metrics.items():
    logger.info(f"{mode_name:<15} {metric['avg_rank']:<12.2f} "
               f"{metric['top3_hit_rate']:<15.1f}% {metric['top1_accuracy']:<12.1f}%")

# 推荐建议
logger.info("\n💡 推荐建议:")

# 找出平均排名最低的
best_avg_rank = min(metrics.items(), key=lambda x: x[1]['avg_rank'])
logger.info(f"  • 追求检索质量: {best_avg_rank[0]} (平均排名: {best_avg_rank[1]['avg_rank']:.2f})")

# 找出 Top-1 准确率最高的
best_top1 = max(metrics.items(), key=lambda x: x[1]['top1_accuracy'])
logger.info(f"  • 追求最佳答案: {best_top1[0]} (Top-1 准确率: {best_top1[1]['top1_accuracy']:.1%})")

# 找出最快的
fastest = min(times, key=lambda x: x[1])
logger.info(f"  • 追求检索速度: {fastest[0]} (耗时: {fastest[1]:.2f}s)")

# 找出 Top-3 命中率最高的
best_top3 = max(metrics.items(), key=lambda x: x[1]['top3_hit_rate'])
logger.info(f"  • 追求稳定性: {best_top3[0]} (Top-3 命中率: {best_top3[1]['top3_hit_rate']:.1f}%)")

logger.info("\n🎯 综合评估:")

# 综合评分
for mode_name, metric in metrics.items():
    # 归一化评分（0-100）
    rank_score = (6 - metric['avg_rank']) / 5 * 40  # 平均排名评分（0-40）
    top3_score = metric['top3_hit_rate'] * 0.4  # Top-3 命中率评分（0-40）
    top1_score = metric['top1_accuracy'] * 20  # Top-1 准确率评分（0-20）

    total_score = rank_score + top3_score + top1_score
    logger.info(f"  {mode_name}: {total_score:.1f}/100")

logger.info("\n" + "=" * 100)
logger.info("测试完成！")
logger.info("=" * 100)
logger.info(f"\n日志已保存到: {log_file}")
