#!/usr/bin/env python3
"""
测试 Amygdala.retrieval() 方法
"""
import sys
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

from workflow.amygdala import Amygdala

print("=" * 80)
print("测试 Amygdala.retrieval() 方法")
print("=" * 80)

# 初始化 Amygdala
print("\n【初始化】创建 Amygdala 实例...")
amygdala = Amygdala(
    save_dir="./test_retrieval_db",
    particle_collection_name="test_particles_retrieval",
    conversation_namespace="test_conversation",
    embedding_model=None,  # 不使用嵌入模型
    auto_link_particles=False
)
print("✓ Amygdala 初始化完成")

# 添加测试数据
print("\n【添加测试数据】添加对话...")
test_conversations = [
    "I love Python programming! It's amazing for data science.",
    "Machine learning is fascinating. I enjoy working with neural networks.",
    "JavaScript is great for web development. React and Vue are popular frameworks.",
    "The weather is beautiful today. I feel happy and energetic.",
    "Java is widely used in enterprise applications. Spring Boot is a popular framework."
]

for i, conv in enumerate(test_conversations, 1):
    print(f"\n添加对话 {i}/{len(test_conversations)}: {conv[:50]}...")
    result = amygdala.add(conv)
    print(f"  ✓ 生成了 {result['particle_count']} 个粒子")

print(f"\n✓ 总共添加了 {len(test_conversations)} 个对话")

# 测试1：粒子检索模式
print("\n" + "=" * 80)
print("【测试1】粒子检索模式 (retrieval_mode='particle')")
print("=" * 80)

query1 = "I enjoy coding with Python"
print(f"\n查询文本: {query1}")

particle_results = amygdala.retrieval(
    query_text=query1,
    retrieval_mode="particle",
    top_k=5
)

print(f"\n检索到 {len(particle_results)} 个相关粒子:")
for i, p in enumerate(particle_results, 1):
    print(f"\n  粒子 {i}:")
    print(f"    - ID: {p['particle_id']}")
    print(f"    - 实体: {p['entity']}")
    print(f"    - 相似度得分: {p['score']:.4f}")
    print(f"    - 匹配类型: {p['match_type']}")
    print(f"    - 所属对话: {p['conversation_id']}")

# 测试2：Chunk检索模式
print("\n" + "=" * 80)
print("【测试2】Chunk检索模式 (retrieval_mode='chunk')")
print("=" * 80)

query2 = "web development and programming"
print(f"\n查询文本: {query2}")

chunk_results = amygdala.retrieval(
    query_text=query2,
    retrieval_mode="chunk",
    top_k=3
)

print(f"\n检索到 {len(chunk_results)} 个相关对话片段:")
for i, chunk in enumerate(chunk_results, 1):
    print(f"\n  Chunk {i} (Rank: {chunk['rank']}):")
    print(f"    - 对话 ID: {chunk['conversation_id']}")
    print(f"    - 得分: {chunk['score']:.1f}")
    print(f"    - 包含粒子数: {chunk['particle_count']}")
    print(f"    - 粒子 IDs: {chunk['particle_ids']}")
    print(f"    - 文本: {chunk['text'][:100]}{'...' if len(chunk['text']) > 100 else ''}")

# 测试3：验证chunk排序规则
print("\n" + "=" * 80)
print("【测试3】验证 Chunk 排序规则")
print("=" * 80)
print("\n排序规则说明:")
print("  - 一个 chunk 的得分取决于它包含的检索到的粒子")
print("  - 包含的越靠前的粒子（在搜索结果中位置越靠前）越多，得分越高")
print(f"  - 计算公式: chunk_score = sum((total_particles - position) for each particle in chunk)")

# 查询一个相关的文本
query3 = "Python and machine learning"
print(f"\n查询文本: {query3}")

chunk_results2 = amygdala.retrieval(
    query_text=query3,
    retrieval_mode="chunk",
    top_k=5
)

print(f"\n检索结果（按得分降序排列）:")
for i, chunk in enumerate(chunk_results2, 1):
    print(f"\n  Rank {i}: Score={chunk['score']:.1f}, Particles={chunk['particle_count']}")
    print(f"    文本: {chunk['text'][:80]}...")

# 验证排序是否正确
if len(chunk_results2) >= 2:
    scores = [c['score'] for c in chunk_results2]
    is_sorted = scores == sorted(scores, reverse=True)
    print(f"\n✓ 排序验证: {'正确' if is_sorted else '错误'}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
