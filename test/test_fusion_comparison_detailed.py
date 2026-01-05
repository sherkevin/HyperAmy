#!/usr/bin/env python3
"""
融合检索对比测试 - 详细版

输出每个检索模块的完整上下文列表，便于对比分析
"""

import logging
import sys
import time
from pathlib import Path

# 设置日志文件
log_file = Path("./log/test_fusion_comparison_detailed.log")
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
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

print("=" * 120)
print("融合检索对比测试（详细版）：GraphFusion vs HippoRAG vs Amygdala")
print("=" * 120)

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

# 使用已有的数据库初始化
print("\n" + "=" * 120)
print("【使用已有数据库初始化系统】")
print("=" * 120)

# 融合检索器
print("\n[1/3] 加载图谱融合检索器...")
fusion = GraphFusionRetriever(
    amygdala_save_dir="./test_graph_fusion_amygdala_db",
    hipporag_save_dir="./test_graph_fusion_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    auto_link_particles=False
)
print("✓ 图谱融合检索器加载完成")

# HippoRAG
print("\n[2/3] 加载 HippoRAG...")
hipporag = HippoRAGWrapper(
    save_dir="./test_graph_fusion_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    llm_base_url=BASE_URL,
    embedding_model_name=f"VLLM/{DEFAULT_EMBEDDING_MODEL}",
    embedding_base_url=API_URL_EMBEDDINGS
)
print("✓ HippoRAG 加载完成")

# Amygdala
print("\n[3/3] 加载 Amygdala...")
amygdala = Amygdala(
    save_dir="./test_graph_fusion_amygdala_db",
    particle_collection_name="fusion_particles",
    conversation_namespace="fusion",
    auto_link_particles=False
)
print("✓ Amygdala 加载完成")

# 辅助函数
def get_chunk_type(chunk_text):
    for i, chunk in enumerate(chunks):
        if chunk_text == chunk:
            return chunk_types[i]
    return "Unknown"

def format_chunk_output(rank, text, score, score_type="分数"):
    """格式化单个chunk的输出"""
    chunk_type = get_chunk_type(text)
    output = f"\n{'=' * 120}\n"
    output += f"Rank {rank}: {chunk_type}\n"
    output += f"{'=' * 120}\n"
    output += f"{score_type}: {score}\n"
    output += f"{'─' * 120}\n"
    output += f"{text}\n"
    output += f"{'=' * 120}"
    return output

# ========== 测试 1: 图谱融合检索 ==========
print("\n" + "=" * 120)
print("【测试 1】图谱融合检索（GraphFusion: HippoRAG + Amygdala）")
print("=" * 120)
print(f"\nQuery: {query}")
print("\n权重配置: emotion=0.3, semantic=0.5, fact=0.2")

start_time = time.time()
fusion_results = fusion.retrieve(
    query=query,
    top_k=5,
    emotion_weight=0.3,
    semantic_weight=0.5,
    fact_weight=0.2
)
fusion_time = time.time() - start_time

print("\n" + "=" * 120)
print(f"【图谱融合检索结果】（检索耗时: {fusion_time:.2f}s）")
print("=" * 120)

for result in fusion_results:
    print(format_chunk_output(result['rank'], result['text'], result['score'], "PPR分数"))

# ========== 测试 2: HippoRAG 单独检索 ==========
print("\n\n" + "=" * 120)
print("【测试 2】HippoRAG 单独检索")
print("=" * 120)
print(f"\nQuery: {query}")

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

print("\n" + "=" * 120)
print(f"【HippoRAG 检索结果】（检索耗时: {hipporag_time:.2f}s）")
print("=" * 120)

for result in hipporag_results:
    print(format_chunk_output(result['rank'], result['text'], result['score'], "PPR分数"))

# ========== 测试 3: Amygdala 单独检索 ==========
print("\n\n" + "=" * 120)
print("【测试 3】Amygdala 单独检索（双曲空间情绪相似度检索）")
print("=" * 120)
print(f"\nQuery: {query}")

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
        top_k=10,  # 增加top_k以获得更多结果
        cone_width=100  # 放宽cone_width
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

        # 去重：每个conversation只保留最相关的结果
        seen_conv_ids = set()
        for result in search_results:
            conv_id = result.metadata.get("conversation_id", "")
            if conv_id and conv_id not in seen_conv_id:
                chunk_text = conv_to_text.get(conv_id, "")
                if chunk_text:
                    amygdala_results.append({
                        'rank': len(amygdala_results) + 1,
                        'text': chunk_text,
                        'score': result.score
                    })
                    seen_conv_id = conv_id

            if len(amygdala_results) >= 5:
                break

amygdala_time = time.time() - start_time

print("\n" + "=" * 120)
print(f"【Amygdala 检索结果】（检索耗时: {amygdala_time:.2f}s）")
print("=" * 120)

if amygdala_results:
    for result in amygdala_results:
        print(format_chunk_output(result['rank'], result['text'], result['score'], "双曲距离（越小越相似）"))
else:
    print("未检索到结果（可能需要调整 cone_width 或其他参数）")

# ========== 详细对比分析 ==========
print("\n\n" + "=" * 120)
print("【详细对比分析】")
print("=" * 120)

# 关键 chunks
key_chunks = {
    "情感对峙": 4,
    "东方哲学": 3,
    "拒绝葡萄干": 2,
    "药丸场景": 1,
    "早餐场景": 0
}

all_modes = [
    ("图谱融合", fusion_results),
    ("HippoRAG", hipporag_results),
    ("Amygdala", amygdala_results)
]

def get_rank(results, chunk_idx):
    for result in results:
        for i, chunk in enumerate(chunks):
            if result.get('text') == chunk and i == chunk_idx:
                return result['rank']
    return None

# 1. 排名对比表格
print("\n" + "=" * 120)
print("1. 排名对比表格")
print("=" * 120)

print(f"\n{'检索方式':<20} {'情感对峙':<15} {'东方哲学':<15} {'拒绝葡萄干':<15} {'药丸场景':<15} {'早餐场景':<15}")
print("─" * 110)

for mode_name, results in all_modes:
    row = f"{mode_name:<20}"
    for chunk_name, chunk_idx in key_chunks.items():
        rank = get_rank(results, chunk_idx)
        rank_str = f"Rank {rank}" if rank else "未检索到"
        row += f"{rank_str:<15}"
    print(row)

# 2. 平均排名统计
print("\n" + "=" * 120)
print("2. 平均排名统计（越低越好）")
print("=" * 120)

print(f"\n{'检索方式':<20} {'平均排名':<15} {'标准差':<15} {'检索到的chunks':<20}")
print("─" * 80)

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
        retrieved_count = len(results)
        print(f"{mode_name:<20} {avg_rank:<15.2f} {std_dev:<15.2f} {retrieved_count}/{len(chunks)}")
    else:
        print(f"{mode_name:<20} {'-':<15} {'-':<15} {len(results)}/{len(chunks)}")

# 3. Top-3 命中率（关键chunks）
print("\n" + "=" * 120)
print("3. Top-3 命中率（关键 chunks：情感对峙、东方哲学、拒绝葡萄干）")
print("=" * 120)

key_chunk_indices = [4, 3, 2]

print(f"\n{'检索方式':<20} {'命中数':<15} {'命中率':<15} {'详情':<50}")
print("─" * 110)

for mode_name, results in all_modes:
    hits = []
    for chunk_idx in key_chunk_indices:
        rank = get_rank(results, chunk_idx)
        if rank:
            chunk_name = list(key_chunks.keys())[list(key_chunks.values()).index(chunk_idx)]
            hits.append(f"{chunk_name}=R{rank}")

    top_3_hits = sum(1 for chunk_idx in key_chunk_indices
                     if (rank := get_rank(results, chunk_idx)) and rank <= 3)
    hit_rate = top_3_hits / len(key_chunk_indices) * 100

    hits_str = ", ".join(hits) if hits else "无"
    print(f"{mode_name:<20} {top_3_hits}/{len(key_chunk_indices)}{'':<8} {hit_rate:<15.1f}% {hits_str:<50}")

# 4. Top-1 准确率（情感对峙 - 最关键答案）
print("\n" + "=" * 120)
print("4. Top-1 准确率（情感对峙 - 最关键答案）")
print("=" * 120)

target_chunk_idx = 4  # 情感对峙

print(f"\n{'检索方式':<20} {'排名':<15} {'评价':<50}")
print("─" * 90)

for mode_name, results in all_modes:
    rank = get_rank(results, target_chunk_idx)
    if rank == 1:
        print(f"{mode_name:<20} Rank {rank:<10} ✓✓✓ 完美！最关键答案排在第一位")
    elif rank:
        deviation = rank - 1
        print(f"{mode_name:<20} Rank {rank:<10} 偏差 {rank-1} 位（{deviation if deviation <= 2 else deviation}位偏离）")
    else:
        print(f"{mode_name:<20} {'-':<15} ✗ 未检索到最关键答案")

# 5. 检索性能对比
print("\n" + "=" * 120)
print("5. 检索性能对比")
print("=" * 120)

times = [
    ("图谱融合", fusion_time),
    ("HippoRAG", hipporag_time),
    ("Amygdala", amygdala_time)
]

min_time = min(t[1] for t in times)

print(f"\n{'检索方式':<20} {'耗时(s)':<15} {'相对倍数':<15} {'性能评价':<30}")
print("─" * 90)

for mode_name, time_cost in times:
    relative = time_cost / min_time
    if relative <= 1.5:
        perf_eval = "优秀"
    elif relative <= 3:
        perf_eval = "良好"
    elif relative <= 10:
        perf_eval = "一般"
    else:
        perf_eval = "较慢"
    print(f"{mode_name:<20} {time_cost:<15.2f} {relative:<15.2f}x {perf_eval:<30}")

# ========== 总结分析 ==========
print("\n\n" + "=" * 120)
print("【总结分析】")
print("=" * 120)

print("\n📊 检索效果总结:")

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
            'top1_accuracy': top_1_acc,
            'retrieved_count': len(results)
        }

print(f"\n{'检索方式':<20} {'平均排名':<12} {'Top-3命中率':<15} {'Top-1准确率':<12} {'检索到chunks':<15}")
print("─" * 90)

for mode_name, metric in metrics.items():
    print(f"{mode_name:<20} {metric['avg_rank']:<12.2f} "
               f"{metric['top3_hit_rate']:<15.1f}% {metric['top1_accuracy']:<12.1f}% "
               f"{metric['retrieved_count']}/{len(chunks)}")

# 推荐建议
print("\n[使用建议]")

best_avg_rank = min(metrics.items(), key=lambda x: x[1]['avg_rank']) if metrics else None
if best_avg_rank:
    print(f"  • 追求检索质量（平均排名最优）: {best_avg_rank[0]} (平均排名: {best_avg_rank[1]['avg_rank']:.2f})")

best_top1 = max(metrics.items(), key=lambda x: x[1]['top1_accuracy']) if metrics else None
if best_top1:
    print(f"  • 追求最佳答案（Top-1准确率最高）: {best_top1[0]} (Top-1 准确率: {best_top1[1]['top1_accuracy']:.1%})")

fastest = min(times, key=lambda x: x[1])
print(f"  • 追求检索速度: {fastest[0]} (耗时: {fastest[1]:.2f}s)")

best_top3 = max(metrics.items(), key=lambda x: x[1]['top3_hit_rate']) if metrics else None
if best_top3:
    print(f"  • 追求稳定性（Top-3命中率最高）: {best_top3[0]} (Top-3 命中率: {best_top3[1]['top3_hit_rate']:.1f}%)")

# 综合评分
print("\n🎯 综合评分（满分100）:")

for mode_name, metric in metrics.items():
    # 归一化评分（0-100）
    rank_score = (6 - metric['avg_rank']) / 5 * 40  # 平均排名评分（0-40）
    top3_score = metric['top3_hit_rate'] * 0.4  # Top-3 命中率评分（0-40）
    top1_score = metric['top1_accuracy'] * 20  # Top-1 准确率评分（0-20）

    total_score = rank_score + top3_score + top1_score
    print(f"  {mode_name}: {total_score:.1f}/100")

print("\n" + "=" * 120)
print("测试完成！")
print("=" * 120)
print(f"\n详细日志已保存到: {log_file}")
print("\n主要结论已在上面的输出中展示，包括三个检索模块的完整上下文列表和详细对比分析。")
