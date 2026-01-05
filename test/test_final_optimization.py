#!/usr/bin/env python3
"""
综合性能测试：方案五 + 方案六A

测试目标：
1. 测试方案五（轻量级NER）的性能
2. 测试方案六A（批量Prompt）的性能
3. 测试综合效果（方案五 + 方案六A）
4. 对比原始版本
"""

import logging
import sys
import time
from pathlib import Path

# 设置日志
log_file = Path("./log/test_final_optimization.log")
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

from particle.emotion_v2 import EmotionV2

print("=" * 120)
print("综合性能测试：方案五 + 方案六A")
print("=" * 120)

# 测试数据
test_text = '''Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate.
"You refuse?" she whispered, her voice full of tears. "Is it because you are our enemy?
To refuse to break bread... means you bring death to this house.'''

test_text_id = "test_comprehensive_optimization"

print(f"\n测试文本长度: {len(test_text)} 字符")

# ========== 配置1：原始版本（baseline，无优化） ==========
print("\n" + "=" * 120)
print("【配置 1】原始版本（baseline）")
print("  - NER: LLM")
print("  - 情感描述: 并行LLM（5 workers）")
print("  - Embedding: 批量API")
print("  - 缓存: 禁用")
print("=" * 120)

start_time = time.time()
emotion_v1 = EmotionV2(
    enable_cache=False,
    use_batch_prompt=False  # 使用并行模式
)
nodes_v1 = emotion_v1.process(text=test_text, text_id=f"{test_text_id}_v1")
time_v1 = time.time() - start_time

print(f"\n✓ 配置1完成 (耗时: {time_v1:.2f}s)")
print(f"  成功生成节点: {len(nodes_v1)}")

# ========== 配置2：仅方案六A（批量Prompt） ==========
print("\n" + "=" * 120)
print("【配置 2】仅方案六A（批量Prompt）")
print("  - NER: LLM")
print("  - 情感描述: 批量Prompt（1次LLM调用）")
print("  - Embedding: 批量API")
print("  - 缓存: 禁用")
print("=" * 120)

start_time = time.time()
emotion_v2 = EmotionV2(
    enable_cache=False,
    use_batch_prompt=True  # 使用批量Prompt
)
nodes_v2 = emotion_v2.process(text=test_text, text_id=f"{test_text_id}_v2")
time_v2 = time.time() - start_time

print(f"\n✓ 配置2完成 (耗时: {time_v2:.2f}s)")
print(f"  成功生成节点: {len(nodes_v2)}")

# ========== 配置3：方案五 + 方案六A（综合优化） ==========
print("\n" + "=" * 120)
print("【配置 3】方案五 + 方案六A（综合优化）")
print("  - NER: spaCy（轻量级）")
print("  - 情感描述: 批量Prompt（1次LLM调用）")
print("  - Embedding: 批量API")
print("  - 缓存: 禁用")
print("=" * 120)

from utils.ner_lightweight import LightweightNER

try:
    lightweight_ner = LightweightNER()

    start_time = time.time()
    emotion_v3 = EmotionV2(
        entity_extractor=lightweight_ner,  # 使用轻量级NER
        enable_cache=False,
        use_batch_prompt=True
    )
    nodes_v3 = emotion_v3.process(text=test_text, text_id=f"{test_text_id}_v3")
    time_v3 = time.time() - start_time

    print(f"\n✓ 配置3完成 (耗时: {time_v3:.2f}s)")
    print(f"  成功生成节点: {len(nodes_v3)}")
except Exception as e:
    print(f"\n⚠ 配置3失败: {e}")
    time_v3 = None
    nodes_v3 = None

# ========== 配置4：启用缓存（方案三 + 方案六A） ==========
print("\n" + "=" * 120)
print("【配置 4】缓存 + 批量Prompt")
print("  - NER: LLM")
print("  - 情感描述: 批量Prompt（1次LLM调用）")
print("  - Embedding: 批量API")
print("  - 缓存: 启用（首次运行）")
print("=" * 120)

start_time = time.time()
emotion_v4 = EmotionV2(
    enable_cache=True,
    use_batch_prompt=True
)
nodes_v4_first = emotion_v4.process(text=test_text, text_id=f"{test_text_id}_v4_first")
time_v4_first = time.time() - start_time

print(f"\n✓ 配置4首次运行完成 (耗时: {time_v4_first:.2f}s)")
print(f"  成功生成节点: {len(nodes_v4_first)}")

# 第二次运行（缓存命中）
start_time = time.time()
nodes_v4_cached = emotion_v4.process(text=test_text, text_id=f"{test_text_id}_v4_cached")
time_v4_cached = time.time() - start_time

print(f"\n✓ 配置4缓存运行完成 (耗时: {time_v4_cached:.2f}s)")
print(f"  成功生成节点: {len(nodes_v4_cached)}")

# ========== 性能对比 ==========
print("\n" + "=" * 120)
print("【性能对比分析】")
print("=" * 120)

print(f"\n{'配置':<40} {'耗时(s)':<15} {'加速比':<15} {'节点数':<10}")
print("-" * 90)

baseline_time = time_v1

print(f"{'1. 原始版本（baseline）':<40} {time_v1:<15.2f} {'1.0x':<15} {len(nodes_v1):<10}")

if time_v2:
    speedup_v2 = baseline_time / time_v2
    print(f"{'2. 仅批量Prompt（方案六A）':<40} {time_v2:<15.2f} {speedup_v2:<15.2f}x {len(nodes_v2):<10}")

if time_v3:
    speedup_v3 = baseline_time / time_v3
    print(f"{'3. spaCy NER + 批量Prompt（方案五+六A）':<40} {time_v3:<15.2f} {speedup_v3:<15.2f}x {len(nodes_v3):<10}")

if time_v4_first:
    speedup_v4_first = baseline_time / time_v4_first
    print(f"{'4. 缓存+批量Prompt（首次）':<40} {time_v4_first:<15.2f} {speedup_v4_first:<15.2f}x {len(nodes_v4_first):<10}")

if time_v4_cached:
    speedup_v4_cached = baseline_time / time_v4_cached
    print(f"{'4. 缓存+批量Prompt（缓存命中）':<40} {time_v4_cached:<15.2f} {speedup_v4_cached:<15.2f}x {len(nodes_v4_cached):<10}")

# ========== 总结 ==========
print("\n" + "=" * 120)
print("【总结】")
print("=" * 120)

print("\n优化效果:")

if time_v2:
    speedup_v2 = baseline_time / time_v2
    print(f"\n方案六A（批量Prompt）:")
    print(f"  加速比: {speedup_v2:.2f}x")
    print(f"  时间节省: {baseline_time - time_v2:.2f}s")
    print(f"  评估: {'✓ 极其显著' if speedup_v2 >= 2 else '✓ 良好' if speedup_v2 >= 1.5 else '⚠ 一般'}")

if time_v3:
    speedup_v3 = baseline_time / time_v3
    print(f"\n方案五 + 方案六A（综合优化）:")
    print(f"  加速比: {speedup_v3:.2f}x")
    print(f"  时间节省: {baseline_time - time_v3:.2f}s")
    print(f"  评估: {'✓✓✓ 卓越' if speedup_v3 >= 3 else '✓✓ 极其显著' if speedup_v3 >= 2 else '✓ 显著' if speedup_v3 >= 1.5 else '⚠ 一般'}")

if time_v4_cached:
    speedup_v4_cached = baseline_time / time_v4_cached
    print(f"\n方案三 + 方案六A（缓存+批量Prompt，缓存命中）:")
    print(f"  加速比: {speedup_v4_cached:.2f}x")
    print(f"  时间节省: {baseline_time - time_v4_cached:.2f}s")
    print(f"  评估: {'✓✓✓✓ 卓越（缓存效果显著）' if speedup_v4_cached >= 10 else '✓✓✓ 极其显著'}")

print("\n保底时延分析（无缓存时）:")
if time_v3:
    print(f"  原始版本: {time_v1:.2f}s")
    print(f"  优化后（方案五+六A）: {time_v3:.2f}s")
    print(f"  保底性能提升: {baseline_time / time_v3:.2f}x")

print("\n建议:")
if time_v3 and time_v3 < 3:
    print("  ✓✓✓ 保底时延已降至 <3s，达到生产可用标准！")
elif time_v3 and time_v3 < 5:
    print("  ✓✓ 保底时延已降至 <5s，基本可用！")
elif time_v3:
    print("  ⚠ 保底时延仍较高，建议进一步优化")

print("\n" + "=" * 120)
print("测试完成！")
print("=" * 120)
print(f"\n日志已保存到: {log_file}")
