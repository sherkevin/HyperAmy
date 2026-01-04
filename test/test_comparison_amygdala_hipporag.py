#!/usr/bin/env python3
"""
对比测试：Amygdala vs HippoRAG

使用相同的 Monte Cristo 数据和 query，对比两个检索系统的差异：
- Amygdala: 双曲几何 + 粒子系统
- HippoRAG: 知识图谱 + OpenIE + PPR

测试场景：
- Query: "Why did the Count strictly refuse the muscatel grapes..."
- 相同的 chunks
- 对比排序后的上下文差异
"""

import logging
import sys
import os
import time
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s'
)

# 设置环境变量
from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
os.environ["OPENAI_API_KEY"] = API_KEY

from workflow.amygdala import Amygdala
from workflow.hipporag_wrapper import HippoRAGWrapper

print("=" * 100)
print("对比测试：Amygdala vs HippoRAG")
print("=" * 100)

# ========== 测试数据（Monte Cristo 场景）==========
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

query = "Why did the Count strictly refuse the muscatel grapes and any refreshment offered by Madame de Morcerf (Mercedes) during his visit to her house?"

# 定义 chunk 类型用于显示
chunk_types = {
    0: "早餐场景",
    1: "药丸场景",
    2: "拒绝葡萄干",
    3: "东方哲学（核心答案）",
    4: "情感对峙"
}

top_k = 5

# ========== 初始化 Amygdala ==========
print("\n" + "=" * 100)
print("【初始化】Amygdala")
print("=" * 100)

amygdala_start = time.time()
amygdala = Amygdala(
    save_dir="./test_comparison_amygdala_db",
    particle_collection_name="comparison_particles",
    conversation_namespace="comparison"
)
print(f"✓ Amygdala 初始化完成 ({time.time() - amygdala_start:.2f}s)")

# ========== 初始化 HippoRAG ==========
print("\n" + "=" * 100)
print("【初始化】HippoRAG")
print("=" * 100)

hipporag_start = time.time()
hipporag = HippoRAGWrapper(
    save_dir="./test_comparison_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    llm_base_url=BASE_URL,
    embedding_model_name=f"VLLM/{DEFAULT_EMBEDDING_MODEL}",
    embedding_base_url=API_URL_EMBEDDINGS
)
print(f"✓ HippoRAG 初始化完成 ({time.time() - hipporag_start:.2f}s)")

# ========== 添加数据到 Amygdala ==========
print("\n" + "=" * 100)
print("【添加数据】Amygdala")
print("=" * 100)

amygdala_add_start = time.time()
for i, chunk in enumerate(chunks, 1):
    print(f"  添加 Chunk {i}/{len(chunks)}: {chunk_types[i-1]}")
    amygdala.add(chunk)
amygdala_add_time = time.time() - amygdala_add_start
print(f"✓ Amygdala 数据添加完成 ({amygdala_add_time:.2f}s)")

# ========== 添加数据到 HippoRAG ==========
print("\n" + "=" * 100)
print("【添加数据】HippoRAG")
print("=" * 100)

hipporag_add_start = time.time()
result = hipporag.add(chunks)
print(f"  添加了 {result['chunk_count']} 个 chunks")
hipporag_add_time = time.time() - hipporag_add_start
print(f"✓ HippoRAG 数据添加完成 ({hipporag_add_time:.2f}s)")

# ========== 执行检索 ==========
print("\n" + "=" * 100)
print("【检索测试】相同 query，对比结果")
print("=" * 100)
print(f"\nQuery: {query}")
print(f"Top-K: {top_k}")

# ========== Amygdala 检索 ==========
print("\n" + "-" * 100)
print("【Amygdala 检索】")
print("-" * 100)

amygdala_retrieve_start = time.time()
amygdala_results = amygdala.retrieval(
    query_text=query,
    retrieval_mode="chunk",
    top_k=top_k
)
amygdala_retrieve_time = time.time() - amygdala_retrieve_start

print(f"\n检索时间: {amygdala_retrieve_time:.2f}s")
print(f"检索到 {len(amygdala_results)} 个 chunks:\n")

for result in amygdala_results:
    chunk_type = "Unknown"
    for i, chunk in enumerate(chunks):
        if result['text'] == chunk:
            chunk_type = chunk_types[i]
            break

    print(f"  Rank {result['rank']}: {chunk_type}")
    print(f"    - 得分: {result['score']:.2f}")
    print(f"    - 文本: {result['text'][:100]}...")
    print()

# ========== HippoRAG 检索 ==========
print("-" * 100)
print("【HippoRAG 检索】")
print("-" * 100)

hipporag_retrieve_start = time.time()
hipporag_results = hipporag.retrieve(
    query=query,
    top_k=top_k
)
hipporag_retrieve_time = time.time() - hipporag_retrieve_start

print(f"\n检索时间: {hipporag_retrieve_time:.2f}s")
print(f"检索到 {len(hipporag_results)} 个 chunks:\n")

for result in hipporag_results:
    chunk_type = "Unknown"
    for i, chunk in enumerate(chunks):
        if result['text'] == chunk:
            chunk_type = chunk_types[i]
            break

    print(f"  Rank {result['rank']}: {chunk_type}")
    print(f"    - 得分: {result['score']:.4f}")
    print(f"    - 文本: {result['text'][:100]}...")
    print()

# ========== 对比分析 ==========
print("\n" + "=" * 100)
print("【对比分析】")
print("=" * 100)

# 创建 chunk 到排名的映射
amygdala_ranking = {}
for result in amygdala_results:
    for i, chunk in enumerate(chunks):
        if result['text'] == chunk:
            amygdala_ranking[i] = result['rank']
            break

hipporag_ranking = {}
for result in hipporag_results:
    for i, chunk in enumerate(chunks):
        if result['text'] == chunk:
            hipporag_ranking[i] = result['rank']
            break

print("\n1. 排名对比:")
print(f"{'Chunk':<20} {'Amygdala Rank':<20} {'HippoRAG Rank':<20} {'差异'}")
print("-" * 80)

for i, chunk_type in chunk_types.items():
    amygdala_rank = amygdala_ranking.get(i, "-")
    hipporag_rank = hipporag_ranking.get(i, "-")

    if amygdala_rank != "-" and hipporag_rank != "-":
        diff = abs(amygdala_rank - hipporag_rank)
        diff_str = f"{diff:+d}" if diff != 0 else "相同"
        indicator = "⚠️" if diff > 2 else ("✓" if diff == 0 else "")
    elif amygdala_rank == "-":
        diff_str = "仅 HippoRAG"
        indicator = "→"
    else:
        diff_str = "仅 Amygdala"
        indicator = "←"

    print(f"{chunk_type:<20} {str(amygdala_rank):<20} {str(hipporag_rank):<20} {diff_str} {indicator}")

# 计算重叠度
print("\n2. Top-{0} 重叠度:".format(top_k))
amygdala_top_k_chunks = set()
for result in amygdala_results:
    for i, chunk in enumerate(chunks):
        if result['text'] == chunk:
            amygdala_top_k_chunks.add(i)
            break

hipporag_top_k_chunks = set()
for result in hipporag_results:
    for i, chunk in enumerate(chunks):
        if result['text'] == chunk:
            hipporag_top_k_chunks.add(i)
            break

intersection = amygdala_top_k_chunks & hipporag_top_k_chunks
union = amygdala_top_k_chunks | hipporag_top_k_chunks

overlap_rate = len(intersection) / len(union) if union else 0
print(f"  - Amygdala 独有: {amygdala_top_k_chunks - hipporag_top_k_chunks}")
print(f"  - HippoRAG 独有: {hipporag_top_k_chunks - amygdala_top_k_chunks}")
print(f"  - 两者共有: {intersection}")
print(f"  - 重叠率: {overlap_rate:.1%}")

# 性能对比
print("\n3. 性能对比:")
print(f"  - Amygdala 添加时间: {amygdala_add_time:.2f}s")
print(f"  - HippoRAG 添加时间: {hipporag_add_time:.2f}s")
print(f"  - Amygdala 检索时间: {amygdala_retrieve_time:.2f}s")
print(f"  - HippoRAG 检索时间: {hipporag_retrieve_time:.2f}s")

# 系统特征
print("\n4. 系统特征:")
print(f"  - Amygdala: 双曲几何 + 粒子系统")
print(f"    - 存储路径: ./test_comparison_amygdala_db")
print(f"    - 检索到的 chunks: {len(amygdala_results)}")

print(f"    - 粒子数: {len(amygdala.particle_to_conversation)}")

print(f"\n  - HippoRAG: 知识图谱 + OpenIE + PPR")
print(f"    - 存储路径: ./test_comparison_hipporag_db")
print(f"    - 检索到的 chunks: {len(hipporag_results)}")

hipporag_stats = hipporag.get_stats()
print(f"    - 图谱节点: {hipporag_stats['graph_nodes']}")
print(f"    - 图谱边: {hipporag_stats['graph_edges']}")
print(f"    - 实体数: {hipporag_stats['entities']}")
print(f"    - 事实数: {hipporag_stats['facts']}")

# ========== 关键发现 ==========
print("\n" + "=" * 100)
print("【关键发现】")
print("=" * 100)

# 检查是否检索到核心答案（Chunk 4 - 东方哲学）
chunk_4_amygdala = amygdala_ranking.get(3, "-")
chunk_4_hipporag = hipporag_ranking.get(3, "-")

print("\n1. 核心答案 Chunk（东方哲学）:")
print(f"   - Amygdala 排名: {chunk_4_amygdala}")
print(f"   - HippoRAG 排名: {chunk_4_hipporag}")

if chunk_4_amygdala != "-" and chunk_4_hipporag != "-":
    if chunk_4_amygdala < chunk_4_hipporag:
        print(f"   → Amygdala 更准确地找到了核心答案")
    elif chunk_4_hipporag < chunk_4_amygdala:
        print(f"   → HippoRAG 更准确地找到了核心答案")
    else:
        print(f"   → 两个系统排名相同")

# 检查是否检索到情感对峙（Chunk 5）
chunk_5_amygdala = amygdala_ranking.get(4, "-")
chunk_5_hipporag = hipporag_ranking.get(4, "-")

print("\n2. 情感对峙 Chunk:")
print(f"   - Amygdala 排名: {chunk_5_amygdala}")
print(f"   - HippoRAG 排名: {chunk_5_hipporag}")

print("\n" + "=" * 100)
print("测试完成！")
print("=" * 100)
