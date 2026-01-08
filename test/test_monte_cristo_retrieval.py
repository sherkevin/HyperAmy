#!/usr/bin/env python3
"""
真实案例测试：基督山伯爵拒绝葡萄干场景

测试场景：
- Query: "Why did the Count strictly refuse the muscatel grapes and any refreshment
         offered by Madame de Morcerf (Mercedes) during his visit to her house?"
- 存储的Chunk: 来自《基督山伯爵》的几个场景
- 检索模式: chunk
- 目标: 检索到相关的上下文，揭示Count拒绝食物的真正原因
"""
import sys
import logging
from typing import List

# 设置详细的日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s'
)

from workflow.amygdala import Amygdala

print("=" * 100)
print("真实案例测试：基督山伯爵 - 拒绝葡萄干场景")
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

for i, chunk in enumerate(chunks, 1):
    print(f"\n添加 Chunk {i}/{len(chunks)}:")
    print(f"  预览: {chunk[:100]}...")

    result = amygdala.add(chunk)
    print(f"  ✓ 生成了 {result['particle_count']} 个粒子")
    print(f"  ✓ 对话ID: {result['conversation_id']}")

print(f"\n✓ 总共添加了 {len(chunks)} 个chunk，生成了 {sum(amygdala.add(c)['particle_count'] for c in chunks)} 个粒子")

# 测试查询
query = "Why did the Count strictly refuse the muscatel grapes and any refreshment offered by Madame de Morcerf (Mercedes) during his visit to her house?"

print("\n" + "=" * 100)
print("【检索测试】执行 Chunk 模式检索")
print("=" * 100)
print(f"\n查询问题: {query}")
print(f"查询长度: {len(query)} 字符")
print(f"检索模式: chunk")
print(f"期望结果: 检索到Chunk 5（东方哲学）和Chunk 6（情感对峙）")

# 执行检索
print("\n开始检索...")
results = amygdala.retrieval(
    query_text=query,
    retrieval_mode="chunk",
    top_k=5,
    cone_width=50
)

# 详细打印检索结果
print("\n" + "=" * 100)
print("【检索结果】详细分析")
print("=" * 100)

if not results:
    print("❌ 未检索到任何结果")
    sys.exit(1)

print(f"\n✓ 检索到 {len(results)} 个相关chunk:\n")

for rank, chunk_result in enumerate(results, 1):
    print(f"\n{'=' * 100}")
    print(f"【结果 #{rank}】Rank: {chunk_result['rank']}")
    print(f"{'=' * 100}")

    print(f"\n📊 评分信息:")
    print(f"  - Chunk ID: {chunk_result['conversation_id']}")
    print(f"  - 得分: {chunk_result['score']:.1f}")
    print(f"  - 包含粒子数: {chunk_result['particle_count']}")
    print(f"  - 粒子列表: {chunk_result['particle_ids']}")

    print(f"\n📝 完整文本:")
    print(f"  {chunk_result['text']}")

    # 分析这个chunk与查询的相关性
    print(f"\n🔍 相关性分析:")

    text_lower = chunk_result['text'].lower()
    query_lower = query.lower()

    # 提取关键词
    keywords = []
    if "count" in text_lower or "monte cristo" in text_lower:
        keywords.append("提到Count/Monte Cristo")
    if "eat" in text_lower or "refuse" in text_lower or "bread" in text_lower:
        keywords.append("涉及进食/拒绝")
    if "mercedes" in text_lower:
        keywords.append("提到Mercedes")
    if "east" in text_lower or "revenge" in text_lower or "kill" in text_lower:
        keywords.append("涉及东方/复仇")
    if "terror" in text_lower or "tears" in text_lower or "avenger" in text_lower:
        keywords.append("情感对峙")

    if keywords:
        print(f"  - 关键特征: {', '.join(keywords)}")
    else:
        print(f"  - 关键特征: 通过语义相似度匹配")

# 分析检索质量
print("\n" + "=" * 100)
print("【检索质量分析】")
print("=" * 100)

# 检查是否检索到关键chunk（Chunk 5 - 东方哲学）
chunk_5_found = False
chunk_6_found = False

for result in results:
    text = result['text']
    if "In the countries of the East" in text or "whom I wish to kill" in text:
        chunk_5_found = True
        print(f"\n✓ 关键Chunk（东方哲学）已找到:")
        print(f"  - 排名: {result['rank']}")
        print(f"  - 得分: {result['score']:.1f}")
        print(f"  - 包含关键信息: 东方关于进食和复仇的哲学")

    if "Mercedes looked at him with terror" in text or "avenger" in text:
        chunk_6_found = True
        print(f"\n✓ 关键Chunk（情感对峙）已找到:")
        print(f"  - 排名: {result['rank']}")
        print(f"  - 得分: {result['score']:.1f}")
        print(f"  - 包含关键信息: Mercedes的恐惧和觉醒")

print("\n" + "=" * 100)
print("【最终结论】")
print("=" * 100)

if chunk_5_found and chunk_6_found:
    print("\n✅ 检索成功！系统成功找到了解释Count拒绝食物原因的关键chunk：")
    print("\n1. Chunk 5 揭示了根本原因：")
    print("   '在东方，人们一起进食就会结成神圣的兄弟情谊。'")
    print("   '因此，我绝不在我想杀的人家里进食。'")
    print("   '如果我分享了他们的面包，我就被荣誉禁止复仇。'")

    print("\n2. Chunk 6 展示了场景的高潮：")
    print("   'Mercedes颤抖着意识到，拒绝共同进食意味着复仇。'")
    print("   '站在她面前的不仅仅是访客，而是一个记得过去的复仇者。'")

    print("\n💡 检索系统成功通过语义理解，找到了答案的核心！")

else:
    print("\n⚠️  部分关键chunk未找到")
    if chunk_5_found:
        print("✓ 找到了东方哲学chunk（核心原因）")
    else:
        print("✗ 未找到东方哲学chunk")

    if chunk_6_found:
        print("✓ 找到了情感对峙chunk")
    else:
        print("✗ 未找到情感对峙chunk")

print("\n" + "=" * 100)
print("测试完成！")
print("=" * 100)
