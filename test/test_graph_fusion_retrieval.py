#!/usr/bin/env python3
"""
图谱融合检索测试 - HippoRAG + Amygdala 实体级融合

测试场景：
1. 统一的实体抽取
2. HippoRAG 语义扩展
3. Amygdala 情绪扩展
4. 实体权重融合
5. PPR 传播
6. 返回排序后的 chunks

使用 Monte Cristo 数据集
"""

import logging
import sys
import time
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s'
)

# 设置环境变量
from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
import os
os.environ["OPENAI_API_KEY"] = API_KEY

from workflow.graph_fusion_retrieval import GraphFusionRetriever

print("=" * 100)
print("图谱融合检索测试：HippoRAG + Amygdala 实体级融合")
print("=" * 100)

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

# ========== 初始化融合检索器 ==========
print("\n" + "=" * 100)
print("【初始化】图谱融合检索器")
print("=" * 100)

fusion = GraphFusionRetriever(
    amygdala_save_dir="./test_graph_fusion_amygdala_db",
    hipporag_save_dir="./test_graph_fusion_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    auto_link_particles=False
)
print("✓ 图谱融合检索器初始化完成")

# ========== 添加数据 ==========
print("\n" + "=" * 100)
print("【添加数据】")
print("=" * 100)

add_start = time.time()
result = fusion.add(chunks)
add_time = time.time() - add_start

print(f"✓ 数据添加完成 ({add_time:.2f}s):")
print(f"  - Amygdala 粒子数: {result['amygdala_count']}")
print(f"  - HippoRAG chunks: {result['hipporag_count']}")
print(f"  - 总 chunks: {result['total_chunks']}")

# ========== 测试 1: 默认权重融合 ==========
print("\n" + "=" * 100)
print("【测试 1】默认权重融合")
print("=" * 100)
print(f"\nQuery: {query}")
print(f"权重配置: emotion=0.3, semantic=0.5, fact=0.2\n")

start_time = time.time()
results_default = fusion.retrieve(
    query=query,
    top_k=5,
    emotion_weight=0.3,
    semantic_weight=0.5,
    fact_weight=0.2
)
default_time = time.time() - start_time

print(f"\n【融合检索结果】（耗时: {default_time:.2f}s）")
print(f"检索到 {len(results_default)} 个 chunks:\n")

for result in results_default:
    # 找到 chunk 类型
    chunk_type = "Unknown"
    for i, chunk in enumerate(chunks):
        if result['text'] == chunk:
            chunk_type = chunk_types[i]
            break

    print(f"  Rank {result['rank']}: {chunk_type}")
    print(f"    - PPR 分数: {result['score']:.4f}")
    print()

# ========== 测试 2: 高情绪权重 ==========
print("\n" + "=" * 100)
print("【测试 2】高情绪权重")
print("=" * 100)
print(f"\nQuery: {query}")
print(f"权重配置: emotion=0.7, semantic=0.2, fact=0.1\n")

start_time = time.time()
results_emotion = fusion.retrieve(
    query=query,
    top_k=5,
    emotion_weight=0.7,
    semantic_weight=0.2,
    fact_weight=0.1
)
emotion_time = time.time() - start_time

print(f"\n【高情绪权重结果】（耗时: {emotion_time:.2f}s）")
print(f"检索到 {len(results_emotion)} 个 chunks:\n")

for result in results_emotion:
    chunk_type = "Unknown"
    for i, chunk in enumerate(chunks):
        if result['text'] == chunk:
            chunk_type = chunk_types[i]
            break

    print(f"  Rank {result['rank']}: {chunk_type}")
    print(f"    - PPR 分数: {result['score']:.4f}")
    print()

# ========== 测试 3: 高语义权重 ==========
print("\n" + "=" * 100)
print("【测试 3】高语义权重")
print("=" * 100)
print(f"\nQuery: {query}")
print(f"权重配置: emotion=0.1, semantic=0.7, fact=0.2\n")

start_time = time.time()
results_semantic = fusion.retrieve(
    query=query,
    top_k=5,
    emotion_weight=0.1,
    semantic_weight=0.7,
    fact_weight=0.2
)
semantic_time = time.time() - start_time

print(f"\n【高语义权重结果】（耗时: {semantic_time:.2f}s）")
print(f"检索到 {len(results_semantic)} 个 chunks:\n")

for result in results_semantic:
    chunk_type = "Unknown"
    for i, chunk in enumerate(chunks):
        if result['text'] == chunk:
            chunk_type = chunk_types[i]
            break

    print(f"  Rank {result['rank']}: {chunk_type}")
    print(f"    - PPR 分数: {result['score']:.4f}")
    print()

# ========== 对比分析 ==========
print("\n" + "=" * 100)
print("【对比分析】")
print("=" * 100)

# 提取排名
def get_rank(results, chunk_idx):
    for result in results:
        for i, chunk in enumerate(chunks):
            if result['text'] == chunk and i == chunk_idx:
                return result['rank']
    return "-"

modes = [
    ("默认权重", results_default),
    ("高情绪权重", results_emotion),
    ("高语义权重", results_semantic)
]

print("\n排名对比:")
print(f"{'Mode':<15} {'情感对峙':<15} {'东方哲学':<15} {'拒绝葡萄干':<15}")
print("-" * 65)

for mode_name, results in modes:
    rank_1 = get_rank(results, 4)  # 情感对峙
    rank_2 = get_rank(results, 3)  # 东方哲学
    rank_3 = get_rank(results, 2)  # 拒绝葡萄干

    # 计算平均排名
    ranks = [r for r in [rank_1, rank_2, rank_3] if r != "-"]
    avg_rank = sum(ranks) / len(ranks) if ranks else "-"

    print(f"{mode_name:<15} {str(rank_1):<15} {str(rank_2):<15} {str(rank_3):<15} (平均: {avg_rank})")

# 性能对比
print("\n性能对比:")
print(f"  {'Mode':<20} {'时间(s)':<15}")
print("-" * 40)
print(f"  {'默认权重':<20} {default_time:<15.2f}")
print(f"  {'高情绪权重':<20} {emotion_time:<15.2f}")
print(f"  {'高语义权重':<20} {semantic_time:<15.2f}")

# ========== 总结 ==========
print("\n" + "=" * 100)
print("【总结】")
print("=" * 100)

print("\n✅ 图谱融合检索优势:")
print("  • 实体级融合：统一在 HippoRAG 图谱中融合语义和情绪信号")
print("  • PPR 传播：利用图谱结构信息")
print("  • 权重可调：可根据场景调整 emotion/semantic/fact 权重")

print("\n📊 融合效果:")
print("  • 语义扩展：基于 HippoRAG 的语义相似度")
print("  • 情绪扩展：基于 Amygdala 的双曲距离")
print("  • Fact 扩展：基于 HippoRAG 的 fact 检索")

print("\n💡 推荐配置:")
print("  • 默认权重: emotion=0.3, semantic=0.5, fact=0.2")
print("  • 追求情绪理解: emotion=0.7, semantic=0.2, fact=0.1")
print("  • 追求语义准确: emotion=0.1, semantic=0.7, fact=0.2")

print("\n" + "=" * 100)
print("测试完成！")
print("=" * 100)
