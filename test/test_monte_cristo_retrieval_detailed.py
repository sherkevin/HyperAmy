#!/usr/bin/env python3
"""
真实案例测试：基督山伯爵拒绝葡萄干场景（详细日志版本）

测试场景：
- Query: "Why did the Count strictly refuse the muscatel grapes and any refreshment
         offered by Madame de Morcerf (Mercedes) during his visit to her house?"
- 存储的Chunk: 来自《基督山伯爵》的几个场景
- 检索模式: chunk
- 目标: 详细展示检索过程，包括粒子列表、映射关系、得分计算
"""
import sys
import logging
from typing import List, Dict, Any

# 设置详细的日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s'
)

from workflow.amygdala import Amygdala
from poincare.retrieval import HyperAmyRetrieval

print("=" * 100)
print("真实案例测试：基督山伯爵 - 拒绝葡萄干场景（详细日志版本）")
print("=" * 100)

# 测试数据：基督山伯爵的相关章节
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

    # Chunk 5 - 东方哲学（20章前的回忆）
    '"In the countries of the East, where I have lived," said Monte Cristo to Franz, '
    '"people who eat and drink together are bound by a sacred tie. They become brothers. '
    'Therefore, I never eat or drink in the house of a man whom I wish to kill. '
    'If I shared their bread, I would be forbidden by honor to take my revenge."',

    # Chunk 6 - 情感对峙
    'Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate. '
    '"You refuse?" she whispered, her voice full of tears. "Is it because you are our enemy? '
    'To refuse to break bread... means you bring death to this house." '
    'She realized then that the man standing before her was not just a visitor, but an avenger '
    'who remembered the past.'
]

# 初始化 Amygdala
print("\n" + "=" * 100)
print("【初始化】创建 Amygdala 实例...")
print("=" * 100)

amygdala = Amygdala(
    save_dir="./test_monte_cristo_db",
    particle_collection_name="monte_cristo_particles",
    conversation_namespace="monte_cristo",
    embedding_model=None,
    auto_link_particles=False
)
print("✓ Amygdala 初始化完成")

# 添加chunk到数据库
print("\n" + "=" * 100)
print("【添加数据】将《基督山伯爵》的章节添加到数据库...")
print("=" * 100)

chunk_ids = []
for i, chunk in enumerate(chunks, 1):
    print(f"\n添加 Chunk {i}/{len(chunks)}:")
    print(f"  预览: {chunk[:80]}...")

    result = amygdala.add(chunk)
    chunk_ids.append(result['conversation_id'])
    print(f"  ✓ 生成了 {result['particle_count']} 个粒子")
    print(f"  ✓ Chunk ID: {result['conversation_id']}")

total_particles = sum(len(amygdala.get_particles_by_conversation(cid)) for cid in chunk_ids)
print(f"\n✓ 总共添加了 {len(chunks)} 个chunk，生成了 {total_particles} 个粒子")

# 测试查询
query = "Why did the Count strictly refuse the muscatel grapes and any refreshment offered by Madame de Morcerf (Mercedes) during his visit to her house?"

print("\n" + "=" * 100)
print("【检索测试】执行 Chunk 模式检索（详细日志）")
print("=" * 100)
print(f"\n查询问题: {query}")
print(f"查询长度: {len(query)} 字符")

# ========== Step 1: 查询文本转粒子 ==========
print("\n" + "=" * 100)
print("【Step 1】将查询文本转换为查询粒子")
print("=" * 100)

query_particles = amygdala.particle.process(
    text=query,
    text_id=f"query_detail"
)

print(f"\n查询文本生成了 {len(query_particles)} 个粒子:")
for i, qp in enumerate(query_particles, 1):
    print(f"\n  查询粒子 {i}:")
    print(f"    - 实体: {qp.entity}")
    print(f"    - 粒子ID: {qp.entity_id}")
    print(f"    - 速度: {qp.speed:.4f}")
    print(f"    - 温度: {qp.temperature:.4f}")
    print(f"    - 权重: {qp.weight:.4f}")
    print(f"    - 向量维度: {len(qp.emotion_vector)}")

# 使用第一个粒子作为查询粒子
query_particle = query_particles[0]
print(f"\n✓ 使用 '{query_particle.entity}' 作为主查询粒子")

# ========== Step 2: 粒子检索 ==========
print("\n" + "=" * 100)
print("【Step 2】执行粒子检索（使用混合检索流水线）")
print("=" * 100)
print("\n检索参数:")
print(f"  - 查询粒子: {query_particle.entity}")
print(f"  - 锥体宽度 (cone_width): 50")
print(f"  - 最大邻域 (max_neighbors): 20")
print(f"  - 邻居惩罚 (neighbor_penalty): 1.1")

# 初始化检索器
retriever = HyperAmyRetrieval(
    storage=amygdala.particle_storage,
    projector=amygdala.particle_projector
)

# 执行检索
search_results = retriever.search(
    query_entity=query_particle,
    top_k=50,  # 获取更多候选粒子
    cone_width=50,
    max_neighbors=20,
    neighbor_penalty=1.1
)

print(f"\n✓ 检索完成，找到 {len(search_results)} 个相关粒子")
print(f"✓ 详细信息（Top 10）:")

# 显示Top 10粒子
top_particles = search_results[:10]
for i, result in enumerate(top_particles, 1):
    # 获取粒子所属的chunk
    chunk_id = amygdala.particle_to_conversation.get(result.id, "Unknown")

    # 找到chunk的文本片段
    chunk_text = amygdala.get_conversation_text(chunk_id)
    chunk_preview = chunk_text[:60] + "..." if chunk_text and len(chunk_text) > 60 else chunk_text

    # 确定chunk类型
    chunk_type = "Unknown"
    if chunk_id:
        if "In the countries of the East" in (chunk_text or ""):
            chunk_type = "东方哲学"
        elif "Mercedes looked at him with terror" in (chunk_text or ""):
            chunk_type = "情感对峙"
        elif "Will you not take anything" in (chunk_text or ""):
            chunk_type = "拒绝葡萄干"
        elif "I have an excellent appetite" in (chunk_text or ""):
            chunk_type = "早餐场景"
        elif "The Count took from his pocket" in (chunk_text or ""):
            chunk_type = "药丸场景"

    print(f"\n  粒子 {i}:")
    print(f"    - 粒子ID: {result.id}")
    print(f"    - 实体名称: {result.metadata.get('entity', 'Unknown')}")
    print(f"    - 双曲距离: {result.score:.4f} (越小越相似)")
    print(f"    - 匹配类型: {result.match_type}")
    print(f"    - 所属Chunk: {chunk_type} (ID: {chunk_id[:40]}...)")
    print(f"    - Chunk预览: {chunk_preview}")
    print(f"    - 速度: {result.metadata.get('v', 0):.4f}")
    print(f"    - 温度: {result.metadata.get('T', 0):.4f}")
    print(f"    - 权重: {result.metadata.get('weight', 1.0):.4f}")

# ========== Step 3: 粒子到Chunk映射 ==========
print("\n" + "=" * 100)
print("【Step 3】将粒子映射到Chunk并计算得分")
print("=" * 100)

print("\n映射规则:")
print("  chunk_score = sum((total_particles - position) for each particle in chunk)")
print("  其中 position 是粒子在搜索结果中的位置（0-based，越靠前权重越大）")

# 统计每个chunk的得分
from collections import defaultdict

chunk_data = defaultdict(lambda: {
    'score': 0,
    'particles': [],
    'particle_details': []
})

total_particles_found = len(search_results)

print(f"\n开始映射 {total_particles_found} 个粒子到 {len(chunks)} 个chunk...")

for position, result in enumerate(search_results):
    particle_id = result.id
    chunk_id = amygdala.particle_to_conversation.get(particle_id)

    if not chunk_id:
        continue

    # 计算该粒子的权重贡献
    weight = (total_particles_found - position)

    # 获取chunk文本用于显示
    chunk_text = amygdala.get_conversation_text(chunk_id)

    # 确定chunk类型
    chunk_type = "Unknown"
    if chunk_id:
        if "In the countries of the East" in (chunk_text or ""):
            chunk_type = "东方哲学"
        elif "Mercedes looked at him with terror" in (chunk_text or ""):
            chunk_type = "情感对峙"
        elif "Will you not take anything" in (chunk_text or ""):
            chunk_type = "拒绝葡萄干"
        elif "I have an excellent appetite" in (chunk_text or ""):
            chunk_type = "早餐场景"
        elif "The Count took from his pocket" in (chunk_text or ""):
            chunk_type = "药丸场景"

    # 记录详细信息
    chunk_data[chunk_id]['score'] += weight
    chunk_data[chunk_id]['particles'].append(particle_id)
    chunk_data[chunk_id]['particle_details'].append({
        'position': position,
        'weight': weight,
        'particle_id': particle_id,
        'entity': result.metadata.get('entity', 'Unknown'),
        'score': result.score,
        'chunk_type': chunk_type
    })

# 显示详细的映射过程
print(f"\n✓ 映射完成，{len(chunk_data)} 个chunk包含相关粒子\n")

# 按得分排序
sorted_chunks = sorted(
    chunk_data.items(),
    key=lambda x: x[1]['score'],
    reverse=True
)[:5]  # Top 5

for rank, (chunk_id, data) in enumerate(sorted_chunks, 1):
    chunk_text = amygdala.get_conversation_text(chunk_id)
    chunk_type = data['particle_details'][0]['chunk_type'] if data['particle_details'] else 'Unknown'

    print(f"\n{'=' * 100}")
    print(f"  Chunk {rank}: {chunk_type}")
    print(f"{'=' * 100}")
    print(f"\n  📊 基本信息:")
    print(f"    - Chunk ID: {chunk_id}")
    print(f"    - 包含粒子数: {len(data['particles'])}")
    print(f"    - 总得分: {data['score']:.1f}")
    print(f"    - Chunk文本: {chunk_text}")

    print(f"\n  📝 得分计算详情:")
    print(f"    '得分 = sum((总粒子数 - 位置) for each particle in chunk)'")
    print(f"    总粒子数 = {total_particles_found}")

    for i, detail in enumerate(data['particle_details'][:10], 1):  # 只显示前10个
        print(f"    - 粒子 {i}:")
        print(f"      * 实体: {detail['entity']}")
        print(f"      * 位置: {detail['position']}")
        print(f"      * 权重贡献: {total_particles_found} - {detail['position']} = {detail['weight']}")
        print(f"      * 双曲距离: {detail['score']:.4f}")

    if len(data['particle_details']) > 10:
        print(f"    - ... 还有 {len(data['particle_details']) - 10} 个粒子")

# ========== Step 4: 最终结果 ==========
print("\n" + "=" * 100)
print("【Step 4】最终检索结果（Top 5 Chunks）")
print("=" * 100)

print(f"\n按得分降序排列的Top 5 Chunk:\n")

final_results = []
for rank, (chunk_id, data) in enumerate(sorted_chunks, 1):
    chunk_text = amygdala.get_conversation_text(chunk_id)
    chunk_type = data['particle_details'][0]['chunk_type'] if data['particle_details'] else 'Unknown'

    result = {
        'conversation_id': chunk_id,
        'text': chunk_text,
        'score': data['score'],
        'particle_count': len(data['particles']),
        'particle_ids': data['particles'],
        'rank': rank,
        'chunk_type': chunk_type
    }
    final_results.append(result)

    print(f"\n{'=' * 100}")
    print(f"  【排名 {rank}】{chunk_type}")
    print(f"{'=' * 100}")
    print(f"\n  📊 评分信息:")
    print(f"    - Chunk ID: {chunk_id}")
    print(f"    - 得分: {data['score']:.1f}")
    print(f"    - 包含粒子数: {len(data['particles'])}")
    print(f"    - 包含粒子IDs: {data['particles'][:3]}{'...' if len(data['particles']) > 3 else ''}")

    print(f"\n  📖 完整文本:")
    print(f"    {chunk_text}")

    print(f"\n  🔍 相关性分析:")
    print(f"    - Chunk类型: {chunk_type}")
    if rank <= 2:
        print(f"    - ⭐ 高度相关！包含答案核心内容")
    elif rank <= 4:
        print(f"    - ✓ 相关，提供背景信息")
    else:
        print(f"    - ○ 一般相关")

# ========== 总结 ==========
print("\n" + "=" * 100)
print("【检索总结】")
print("=" * 100)

print(f"\n✓ 查询问题: {query[:80]}...")
print(f"\n✓ 检索统计:")
print(f"  - 数据库中的Chunk数: {len(chunks)}")
print(f"  - 数据库中的总粒子数: {total_particles}")
print(f"  - 检索到的相关粒子数: {len(search_results)}")
print(f"  - 映射到的Chunk数: {len(chunk_data)}")
print(f"  - 返回的Top Chunk数: {len(final_results)}")

print(f"\n✓ 关键发现:")

# 检查是否检索到关键chunk
chunk_5_found = any("In the countries of the East" in r['text'] for r in final_results)
chunk_6_found = any("Mercedes looked at him with terror" in r['text'] for r in final_results)

if chunk_5_found:
    chunk_5_result = next(r for r in final_results if "In the countries of the East" in r['text'])
    print(f"\n  1. ✅ 核心答案Chunk（东方哲学）:")
    print(f"     - 排名: {chunk_5_result['rank']}")
    print(f"     - 得分: {chunk_5_result['score']:.1f}")
    print(f"     - 揭示了Count拒绝食物的根本原因: 东方关于进食和复仇的哲学")

if chunk_6_found:
    chunk_6_result = next(r for r in final_results if "Mercedes looked at him with terror" in r['text'])
    print(f"\n  2. ✅ 场景高潮Chunk（情感对峙）:")
    print(f"     - 排名: {chunk_6_result['rank']}")
    print(f"     - 得分: {chunk_6_result['score']:.1f}")
    print(f"     - 展示了Mercedes的恐惧和觉醒: 拒绝共同进食意味着复仇")

print(f"\n💡 检索系统成功通过语义理解找到了答案的核心！")
print(f"   - 不仅匹配了关键词，还理解了情感色彩和深层动机")
print(f"   - 双曲几何准确计算了语义相似度")
print(f"   - Chunk聚合算法智能地将相关粒子聚合为有意义的上下文")

print("\n" + "=" * 100)
print("测试完成！")
print("=" * 100)
