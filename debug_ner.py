#!/usr/bin/env python3
"""
NER 实体提取调试脚本

用于测试 HippoRAG 的 NER 功能，找出为什么某些文本无法提取到实体
"""

import logging
from utils.entitiy import Entity

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)


def test_ner_extraction():
    """测试NER实体提取"""

    # 初始化 Entity 提取器
    print("\n" + "=" * 80)
    print("初始化 Entity 提取器")
    print("=" * 80)
    entity_extractor = Entity()

    # 测试用例
    test_cases = [
        # 成功案例
        ("I love Python programming!", "简单Python文本（成功）"),
        ("First conversation about Python.", "简单Python文本（成功）"),

        # 失败案例
        ("I love Python programming and I'm excited about machine learning!", "复合句（失败）"),
        ("The weather is beautiful today, I feel happy.", "情感文本（失败）"),
        ("I'm frustrated with this bug in my code.", "情感文本（失败）"),
        ("Learning new technologies is exciting and challenging.", "抽象概念（失败）"),

        # 边界案例
        ("", "空文本"),
        ("Python programming and machine learning are exciting!", "多个技术名词"),
        ("I love Java, Python, and JavaScript!", "多个编程语言"),
        ("The quick brown fox jumps over the lazy dog.", "无实体文本"),
        ("Google, Microsoft, and Apple are tech giants.", "多个公司名称"),
    ]

    print("\n" + "=" * 80)
    print(f"开始测试 {len(test_cases)} 个文本")
    print("=" * 80)

    results = {
        "success": [],
        "failed": [],
        "unexpected": []
    }

    for idx, (text, description) in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"测试 {idx}/{len(test_cases)}: {description}")
        print(f"{'─' * 80}")
        print(f"文本: {text if text else '(空字符串)'}")
        print(f"长度: {len(text)} 字符")

        try:
            # 提取实体
            entities = entity_extractor.extract_entities(text)

            print(f"\n✓ 提取成功")
            print(f"  实体数量: {len(entities)}")
            print(f"  实体列表: {entities}")

            # 分析结果
            if not text:
                if entities:
                    print(f"  ⚠️  警告: 空文本不应该提取到实体！")
                    results["unexpected"].append((text, description, entities))
                else:
                    print(f"  ✓ 正确: 空文本没有提取到实体")
                    results["success"].append((text, description, entities))
            elif len(entities) > 0:
                print(f"  ✓ 成功提取到实体")
                results["success"].append((text, description, entities))
            else:
                print(f"  ✗ 未提取到实体（可能失败）")
                results["failed"].append((text, description, entities))

        except Exception as e:
            print(f"\n✗ 提取失败")
            print(f"  错误: {str(e)}")
            results["failed"].append((text, description, None))

    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总测试数: {len(test_cases)}")
    print(f"成功: {len(results['success'])}")
    print(f"失败（未提取到实体）: {len(results['failed'])}")
    print(f"意外（空文本提取到实体）: {len(results['unexpected'])}")

    # 详细列出失败的案例
    if results['failed']:
        print("\n" + "-" * 80)
        print("失败的案例（未提取到实体）:")
        print("-" * 80)
        for text, description, entities in results['failed']:
            print(f"  • {description}")
            print(f"    文本: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"    实体: {entities}")

    # 详细列出意外的案例
    if results['unexpected']:
        print("\n" + "-" * 80)
        print("意外的案例（空文本提取到实体）:")
        print("-" * 80)
        for text, description, entities in results['unexpected']:
            print(f"  • {description}")
            print(f"    文本: '{text}'")
            print(f"    实体: {entities}")

    # 详细列出成功的案例
    if results['success']:
        print("\n" + "-" * 80)
        print("成功的案例:")
        print("-" * 80)
        for text, description, entities in results['success']:
            print(f"  • {description}")
            print(f"    文本: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"    实体: {entities}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_ner_extraction()
