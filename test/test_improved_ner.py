#!/usr/bin/env python3
"""
测试改进后的NER Prompt

验证是否能够提取抽象概念
"""
from utils.entitiy import Entity

print("=" * 80)
print("测试改进后的NER Prompt")
print("=" * 80)

# 初始化Entity提取器
entity_extractor = Entity()

# 测试用例 - 之前失败的文本
test_cases = [
    ("The weather is beautiful today.", "天气相关"),
    ("Learning new technologies is exciting and challenging.", "学习技术"),
    ("I'm frustrated with this bug in my code.", "编程问题"),
    ("I love Python programming!", "编程（之前成功）"),
    ("Barack Obama was the 44th president.", "人名（之前成功）"),
    ("", "空文本"),
]

print("\n" + "=" * 80)
print("测试结果")
print("=" * 80)

results = {
    "improved": [],
    "still_failed": []
}

for text, description in test_cases:
    print(f"\n{'─' * 80}")
    print(f"测试: {description}")
    print(f"文本: {repr(text)}")
    print(f"{'─' * 80}")

    if not text:
        print("⚠️  空文本，跳过")
        continue

    try:
        entities = entity_extractor.extract_entities(text)
        print(f"✓ 提取成功")
        print(f"  实体数量: {len(entities)}")
        print(f"  实体列表: {entities}")

        if len(entities) > 0:
            print(f"  ✓✓✓ 改进成功！")
            results["improved"].append((text, description, entities))
        else:
            print(f"  ✗✗✗ 仍然失败")
            results["still_failed"].append((text, description, entities))

    except Exception as e:
        print(f"✗ 提取失败: {e}")
        results["still_failed"].append((text, description, None))

# 打印总结
print("\n" + "=" * 80)
print("测试总结")
print("=" * 80)
print(f"改进成功: {len(results['improved'])}")
print(f"仍然失败: {len(results['still_failed'])}")

if results['improved']:
    print("\n✓ 改进的案例:")
    for text, desc, entities in results['improved']:
        print(f"  • {desc}")
        print(f"    文本: {text}")
        print(f"    实体: {entities}")

if results['still_failed']:
    print("\n✗ 仍然失败的案例:")
    for text, desc, entities in results['still_failed']:
        print(f"  • {desc}")
        print(f"    文本: {text}")
        print(f"    实体: {entities}")

print("\n" + "=" * 80)
