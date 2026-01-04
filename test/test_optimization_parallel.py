#!/usr/bin/env python3
"""
优化方案一测试：并行LLM调用

测试目标：
1. 验证并行化实现的正确性
2. 测量性能提升倍数
3. 确保生成的情感描述质量不变
"""

import logging
import sys
import time
from pathlib import Path

# 设置日志
log_file = Path("./log/test_optimization_parallel.log")
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

from llm.config import API_KEY, BASE_URL
import os
os.environ["OPENAI_API_KEY"] = API_KEY

from utils.sentence import Sentence
from particle.emotion_v2 import EmotionV2

print("=" * 120)
print("优化方案一测试：并行LLM调用")
print("=" * 120)

# 测试数据
test_sentence = '''Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate.
"You refuse?" she whispered, her voice full of tears. "Is it because you are our enemy?
To refuse to break bread... means you bring death to this house."'''

test_entities = ["Mercedes", "terror", "eyes", "hand", "plate", "refuse", "tears", "enemy", "bread", "death"]

print(f"\n测试文本: {test_sentence[:100]}...")
print(f"测试实体数量: {len(test_entities)}")
print(f"测试实体列表: {test_entities}")

# 初始化
print("\n" + "=" * 120)
print("【初始化】")
print("=" * 120)

sentence_processor = Sentence(model_name="DeepSeek-V3.2")
print("✓ Sentence处理器初始化完成")

# ========== 测试1：原始串行版本 ==========
print("\n" + "=" * 120)
print("【测试 1】原始串行版本（baseline）")
print("=" * 120)

start_time = time.time()
descriptions_serial = sentence_processor.generate_affective_descriptions(
    sentence=test_sentence,
    entities=test_entities
)
serial_time = time.time() - start_time

print(f"\n✓ 串行版本完成 (耗时: {serial_time:.2f}s)")
print(f"成功生成: {len([d for d in descriptions_serial.values() if d])}/{len(test_entities)} 个描述")

# 显示部分结果
for entity, desc in list(descriptions_serial.items())[:3]:
    print(f"  {entity}: {desc[:80]}...")

# ========== 测试2：并行版本（方案一） ==========
print("\n" + "=" * 120)
print("【测试 2】并行优化版本（方案一）")
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

# 显示部分结果
for entity, desc in list(descriptions_parallel.items())[:3]:
    print(f"  {entity}: {desc[:80]}...")

# ========== 对比分析 ==========
print("\n" + "=" * 120)
print("【性能对比分析】")
print("=" * 120)

print(f"\n{'版本':<20} {'耗时(s)':<15} {'提升倍数':<15} {'成功率':<15}")
print("-" * 75)

serial_success = len([d for d in descriptions_serial.values() if d]) / len(test_entities) * 100
parallel_success = len([d for d in descriptions_parallel.values() if d]) / len(test_entities) * 100
speedup = serial_time / parallel_time

print(f"{'串行版本':<20} {serial_time:<15.2f} {'1.0x':<15} {serial_success:.1f}%")
print(f"{'并行版本':<20} {parallel_time:<15.2f} {speedup:<15.2f}x {parallel_success:.1f}%")

print("\n" + "=" * 120)
print("【结果验证】")
print("=" * 120)

# 验证生成的描述是否一致
all_match = True
differences = []

for entity in test_entities:
    desc_serial = descriptions_serial.get(entity, "")
    desc_parallel = descriptions_parallel.get(entity, "")

    if desc_serial != desc_parallel:
        all_match = False
        differences.append({
            'entity': entity,
            'serial': desc_serial[:50] if desc_serial else "(empty)",
            'parallel': desc_parallel[:50] if desc_parallel else "(empty)"
        })

if all_match:
    print("\n✓ 验证通过：并行版本生成的描述与串行版本完全一致")
else:
    print(f"\n⚠ 警告：发现 {len(differences)} 个实体生成的描述不一致")
    print("\n不一致的实体:")
    for diff in differences[:5]:
        print(f"  {diff['entity']}:")
        print(f"    串行: {diff['serial']}")
        print(f"    并行: {diff['parallel']}")

print("\n" + "=" * 120)
print("【总结】")
print("=" * 120)

print(f"\n性能提升: {speedup:.2f}x (从 {serial_time:.2f}s 降至 {parallel_time:.2f}s)")
print(f"时间节省: {serial_time - parallel_time:.2f}s")

if speedup >= 3:
    print("\n✓✓✓ 优化效果显著！达到预期目标（3-8倍提升）")
elif speedup >= 2:
    print("\n✓✓ 优化效果良好（2-3倍提升）")
elif speedup >= 1.5:
    print("\n✓ 优化效果一般（1.5-2倍提升）")
else:
    print("\n⚠ 优化效果不明显，可能需要调整参数")

print("\n下一步建议:")
print("  1. 如果速度提升达到预期 → 进入方案二：批量Embedding调用")
print("  2. 如果速度提升未达预期 → 调整 max_workers 参数（当前=5）")

print("\n" + "=" * 120)
print("测试完成！")
print("=" * 120)
print(f"\n日志已保存到: {log_file}")
