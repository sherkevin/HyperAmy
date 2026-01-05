#!/usr/bin/env python3
"""
方案五测试：轻量级 NER 性能测试

测试目标：
1. 对比 LLM vs spaCy 的性能差异
2. 验证实体抽取的准确性
3. 测量加速比
"""

import logging
import sys
import time
from pathlib import Path

# 设置日志
log_file = Path("./log/test_ner_lightweight.log")
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

from llm.config import API_KEY
import os
os.environ["OPENAI_API_KEY"] = API_KEY

from utils.entitiy import Entity
from utils.ner_lightweight import LightweightNER, RuleBasedNER

print("=" * 120)
print("方案五测试：轻量级 NER 性能对比")
print("=" * 120)

# 测试数据
test_queries = [
    "Why did the Count strictly refuse the muscatel grapes and any refreshment offered by Madame de Morcerf (Mercedes) during his visit to her house?",
    "Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate.",
    "The Count of Monte Cristo exacted his revenge on those who had wronged him.",
    "Edmond Dantès was imprisoned in the Château d'If for fourteen years."
]

print(f"\n测试查询数量: {len(test_queries)}")

# 初始化
print("\n" + "=" * 120)
print("【初始化】")
print("=" * 120)

try:
    llm_entity_extractor = Entity(model_name="DeepSeek-V3.2")
    print("✓ LLM Entity 提取器初始化完成")
except Exception as e:
    print(f"⚠ LLM Entity 提取器初始化失败: {e}")
    llm_entity_extractor = None

try:
    lightweight_ner = LightweightNER(model_name="en_core_web_sm")
    print("✓ spaCy NER 初始化完成")
except Exception as e:
    print(f"⚠ spaCy NER 初始化失败: {e}")
    print("  尝试使用规则-based NER...")
    try:
        lightweight_ner = RuleBasedNER()
        print("✓ 规则-based NER 初始化完成（备选方案）")
    except Exception as e2:
        print(f"⚠ 规则-based NER 也初始化失败: {e2}")
        lightweight_ner = None

if not llm_entity_extractor and not lightweight_ner:
    print("\n❌ 所有 NER 系统均初始化失败，无法继续测试")
    sys.exit(1)

# 测试所有查询
all_results = []

for i, query in enumerate(test_queries, 1):
    print("\n" + "=" * 120)
    print(f"【测试 {i}/{len(test_queries)}】")
    print("=" * 120)
    print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

    result = {
        "query": query,
        "llm_entities": None,
        "llm_time": None,
        "lightweight_entities": None,
        "lightweight_time": None
    }

    # LLM NER
    if llm_entity_extractor:
        print("\n[LLM NER]")
        start_time = time.time()
        try:
            llm_entities = llm_entity_extractor.extract_entities(query)
            llm_time = time.time() - start_time
            result["llm_entities"] = llm_entities
            result["llm_time"] = llm_time
            print(f"  耗时: {llm_time:.3f}s")
            print(f"  实体数量: {len(llm_entities)}")
            print(f"  实体列表: {llm_entities}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            result["llm_time"] = None

    # 轻量级 NER
    if lightweight_ner:
        print("\n[轻量级 NER]")
        start_time = time.time()
        try:
            lightweight_entities = lightweight_ner.extract_entities(query)
            lightweight_time = time.time() - start_time
            result["lightweight_entities"] = lightweight_entities
            result["lightweight_time"] = lightweight_time
            print(f"  耗时: {lightweight_time:.3f}s")
            print(f"  实体数量: {len(lightweight_entities)}")
            print(f"  实体列表: {lightweight_entities}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            result["lightweight_time"] = None

    all_results.append(result)

# 性能对比
print("\n" + "=" * 120)
print("【性能对比分析】")
print("=" * 120)

valid_comparisons = [r for r in all_results if r["llm_time"] and r["lightweight_time"]]

if valid_comparisons:
    avg_llm_time = sum(r["llm_time"] for r in valid_comparisons) / len(valid_comparisons)
    avg_lightweight_time = sum(r["lightweight_time"] for r in valid_comparisons) / len(valid_comparisons)
    speedup = avg_llm_time / avg_lightweight_time if avg_lightweight_time > 0 else 0

    print(f"\n{'方法':<20} {'平均耗时(s)':<20} {'加速比':<15}")
    print("-" * 60)
    print(f"{'LLM NER':<20} {avg_llm_time:<20.3f} {'1.0x':<15}")
    print(f"{'轻量级 NER':<20} {avg_lightweight_time:<20.3f} {speedup:<15.2f}x")

    print(f"\n✓ 轻量级 NER 平均加速比: {speedup:.2f}x")
    print(f"✓ 时间节省: {(avg_llm_time - avg_lightweight_time):.3f}s")

# 准确性对比
print("\n" + "=" * 120)
print("【准确性对比分析】")
print("=" * 120)

for i, result in enumerate(all_results, 1):
    if result["llm_entities"] and result["lightweight_entities"]:
        llm_set = set(result["llm_entities"])
        lightweight_set = set(result["lightweight_entities"])

        # 计算重叠
        intersection = llm_set & lightweight_set
        union = llm_set | lightweight_set

        precision = len(intersection) / len(lightweight_set) if lightweight_set else 0
        recall = len(intersection) / len(llm_set) if llm_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n查询 {i}:")
        print(f"  LLM 实体: {sorted(llm_set)}")
        print(f"  轻量级实体: {sorted(lightweight_set)}")
        print(f"  重叠实体: {sorted(intersection)}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")

        # 分析差异
        only_in_llm = llm_set - lightweight_set
        only_in_lightweight = lightweight_set - llm_set

        if only_in_llm:
            print(f"  仅在 LLM 中: {sorted(only_in_llm)}")
        if only_in_lightweight:
            print(f"  仅在轻量级中: {sorted(only_in_lightweight)}")

print("\n" + "=" * 120)
print("【总结】")
print("=" * 120)

if valid_comparisons:
    print(f"\n性能提升: {speedup:.2f}x")
    print(f"时间节省: {(avg_llm_time - avg_lightweight_time):.3f}s / query")

    if speedup >= 20:
        print("\n✓✓✓ 轻量级 NER 性能极其优异！加速超过20倍")
    elif speedup >= 10:
        print("\n✓✓ 轻量级 NER 性能优异！加速超过10倍")
    elif speedup >= 5:
        print("\n✓ 轻量级 NER 性能良好！加速超过5倍")
    else:
        print("\n⚠ 加速效果一般，可能需要优化")

print("\n下一步建议:")
print("  1. 如果准确率可接受 → 进入方案六A：批量Prompt")
print("  2. 如果准确率不足 → 考虑混合策略（LLM + 轻量级）")

print("\n" + "=" * 120)
print("测试完成！")
print("=" * 120)
print(f"\n日志已保存到: {log_file}")
