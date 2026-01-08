#!/usr/bin/env python3
"""
方案六A测试：批量 Prompt 性能测试

测试目标：
1. 对比并行 LLM vs 批量 Prompt 的性能差异
2. 验证情感描述生成的准确性
3. 测量加速比
"""

import logging
import sys
import time
from pathlib import Path

# 设置日志
log_file = Path("./log/test_batch_prompt.log")
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

from utils.sentence import Sentence

print("=" * 120)
print("方案六A测试：批量 Prompt 性能对比")
print("=" * 120)

# 测试数据
test_sentence = '''Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate.
"You refuse?" she whispered, her voice full of tears. "Is it because you are our enemy?
To refuse to break bread... means you bring death to this house.'''

test_entities = ["Mercedes", "terror", "eyes", "hand", "plate", "refuse", "tears", "enemy", "bread", "death"]

print(f"\n测试实体数量: {len(test_entities)}")
print(f"测试实体列表: {test_entities}")

# 初始化
print("\n" + "=" * 120)
print("【初始化】")
print("=" * 120)

sentence_processor = Sentence(model_name="DeepSeek-V3.2")
print("✓ Sentence 处理器初始化完成")

# ========== 测试1：并行版本（baseline） ==========
print("\n" + "=" * 120)
print("【测试 1】并行版本（baseline）")
print("=" * 120)

start_time = time.time()
descriptions_parallel = sentence_processor.generate_affective_descriptions_parallel(
    sentence=test_sentence,
    entities=test_entities,
    max_workers=5
)
parallel_time = time.time() - start_time

print(f"\n✓ 并行版本完成 (耗时: {parallel_time:.2f}s)")
print(f"成功生成: {len([d for d in descriptions_parallel.values() if d])}/{len(test_entities)} 个描述")

# 显示所有结果
for entity, desc in descriptions_parallel.items():
    print(f"  {entity}: {desc}")

# ========== 测试2：批量版本（方案六A） ==========
print("\n" + "=" * 120)
print("【测试 2】批量版本（方案六A）")
print("=" * 120)

start_time = time.time()
descriptions_batch = sentence_processor.generate_affective_descriptions_batch(
    sentence=test_sentence,
    entities=test_entities
)
batch_time = time.time() - start_time

print(f"\n✓ 批量版本完成 (耗时: {batch_time:.2f}s)")
print(f"成功生成: {len([d for d in descriptions_batch.values() if d])}/{len(test_entities)} 个描述")

# 显示所有结果
for entity, desc in descriptions_batch.items():
    print(f"  {entity}: {desc}")

# ========== 对比分析 ==========
print("\n" + "=" * 120)
print("【性能对比分析】")
print("=" * 120)

print(f"\n{'版本':<20} {'耗时(s)':<15} {'加速比':<15} {'成功率':<15}")
print("-" * 75)

parallel_success = len([d for d in descriptions_parallel.values() if d]) / len(test_entities) * 100
batch_success = len([d for d in descriptions_batch.values() if d]) / len(test_entities) * 100
speedup = parallel_time / batch_time if batch_time > 0 else 0

print(f"{'并行版本':<20} {parallel_time:<15.2f} {'1.0x':<15} {parallel_success:.1f}%")
print(f"{'批量版本':<20} {batch_time:<15.2f} {speedup:<15.2f}x {batch_success:.1f}%")

print("\n" + "=" * 120)
print("【准确性对比分析】")
print("=" * 120)

# 验证生成的描述是否合理
print("\n对比每个实体的描述:")
for entity in test_entities:
    desc_parallel = descriptions_parallel.get(entity, "")
    desc_batch = descriptions_batch.get(entity, "")

    print(f"\n{entity}:")
    print(f"  并行: {desc_parallel if desc_parallel else '(empty)'}")
    print(f"  批量: {desc_batch if desc_batch else '(empty)'}")

    # 简单的相似性检查
    if desc_parallel and desc_batch:
        words_parallel = set(desc_parallel.lower().split(', '))
        words_batch = set(desc_batch.lower().split(', '))

        overlap = words_parallel & words_batch
        if overlap:
            similarity = len(overlap) / max(len(words_parallel), len(words_batch)) * 100
            print(f"  相似度: {similarity:.1f}% (重叠词: {', '.join(list(overlap)[:3])})")

print("\n" + "=" * 120)
print("【总结】")
print("=" * 120)

if speedup > 0:
    print(f"\n性能提升: {speedup:.2f}x (从 {parallel_time:.2f}s 降至 {batch_time:.2f}s)")
    print(f"时间节省: {parallel_time - batch_time:.2f}s")

    if speedup >= 5:
        print("\n✓✓✓ 批量Prompt效果极其显著！加速超过5倍")
    elif speedup >= 3:
        print("\n✓✓ 批量Prompt效果显著！加速超过3倍")
    elif speedup >= 1.5:
        print("\n✓ 批量Prompt效果良好（1.5-3倍加速）")
    else:
        print("\n⚠ 批量Prompt效果一般（1.5倍以下加速）")

print("\n关键指标:")
print(f"  LLM 调用次数:")
print(f"    - 并行版本: {len(test_entities)} 次")
print(f"    - 批量版本: 1 次")
print(f"  调用减少: {len(test_entities) - 1} 次")

print("\n下一步建议:")
print("  1. 如果速度提升满意 → 进行综合测试（方案五+六A）")
print("  2. 如果解析有问题 → 优化批量 Prompt 模板")

print("\n" + "=" * 120)
print("测试完成！")
print("=" * 120)
print(f"\n日志已保存到: {log_file}")
