#!/usr/bin/env python3
"""
优化方案三测试：缓存机制

测试目标：
1. 验证缓存机制实现的正确性
2. 测量缓存命中率对性能的影响
3. 确保缓存的数据与新生成的一致
"""

import logging
import sys
import time
import shutil
from pathlib import Path

# 设置日志
log_file = Path("./log/test_optimization_cache.log")
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

from particle.emotion_v2 import EmotionV2

print("=" * 120)
print("优化方案三测试：缓存机制")
print("=" * 120)

# 测试数据
test_sentence = '''Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate.
"You refuse?" she whispered, her voice full of tears. "Is it because you are our enemy?
To refuse to break bread... means you bring death to this house.'''

# 缓存目录
cache_dir = "./test_emotion_cache"

# 清空缓存（如果存在）
if Path(cache_dir).exists():
    shutil.rmtree(cache_dir)
    print(f"\n✓ 清空旧缓存: {cache_dir}")

print(f"\n测试实体数量: 7")
print(f"缓存目录: {cache_dir}")

# 初始化
print("\n" + "=" * 120)
print("【初始化】")
print("=" * 120)

emotion_processor = EmotionV2(enable_cache=True, cache_dir=cache_dir)
print("✓ EmotionV2处理器初始化完成（缓存已启用）")

# ========== 测试1：第一次运行（无缓存，生成所有数据） ==========
print("\n" + "=" * 120)
print("【测试 1】第一次运行（无缓存，baseline）")
print("=" * 120)

text_id_1 = "test_query_1"

start_time = time.time()
nodes_1 = emotion_processor.process(text=test_sentence, text_id=text_id_1)
time_first_run = time.time() - start_time

print(f"\n✓ 第一次运行完成 (耗时: {time_first_run:.2f}s)")
print(f"成功生成: {len(nodes_1)} 个情绪节点")

# 显示部分结果
for i, node in enumerate(list(nodes_1)[:3]):
    print(f"  {i+1}. entity={node.entity}, vector_shape={node.emotion_vector.shape}")

# ========== 测试2：第二次运行（完全命中缓存） ==========
print("\n" + "=" * 120)
print("【测试 2】第二次运行（完全命中缓存）")
print("=" * 120)

text_id_2 = "test_query_2"

start_time = time.time()
nodes_2 = emotion_processor.process(text=test_sentence, text_id=text_id_2)
time_second_run = time.time() - start_time

print(f"\n✓ 第二次运行完成 (耗时: {time_second_run:.2f}s)")
print(f"成功生成: {len(nodes_2)} 个情绪节点")

# 显示部分结果
for i, node in enumerate(list(nodes_2)[:3]):
    print(f"  {i+1}. entity={node.entity}, vector_shape={node.emotion_vector.shape}")

# ========== 测试3：第三次运行（部分新实体） ==========
print("\n" + "=" * 120)
print("【测试 3】第三次运行（部分新实体）")
print("=" * 120)

# 部分修改文本（保留大部分实体）
test_sentence_partial = '''Mercedes looked at him with terror in her eyes. Her hand trembled.
"You refuse?" she whispered with tears. "Is it because you are our enemy?
To refuse to break bread... means you bring death to this house.'''

text_id_3 = "test_query_3"

start_time = time.time()
nodes_3 = emotion_processor.process(text=test_sentence_partial, text_id=text_id_3)
time_third_run = time.time() - start_time

print(f"\n✓ 第三次运行完成 (耗时: {time_third_run:.2f}s)")
print(f"成功生成: {len(nodes_3)} 个情绪节点")

# ========== 对比分析 ==========
print("\n" + "=" * 120)
print("【性能对比分析】")
print("=" * 120)

print(f"\n{'运行':<20} {'耗时(s)':<15} {'提升倍数':<15} {'节点数':<15}")
print("-" * 75)

speedup_2 = time_first_run / time_second_run if time_second_run > 0 else 0
speedup_3 = time_first_run / time_third_run if time_third_run > 0 else 0

print(f"{'第一次运行（无缓存）':<20} {time_first_run:<15.2f} {'1.0x':<15} {len(nodes_1):<15}")
print(f"{'第二次运行（全缓存）':<20} {time_second_run:<15.2f} {speedup_2:<15.2f}x {len(nodes_2):<15}")
print(f"{'第三次运行（部分缓存）':<20} {time_third_run:<15.2f} {speedup_3:<15.2f}x {len(nodes_3):<15}")

print("\n" + "=" * 120)
print("【缓存统计】")
print("=" * 120)

cache_stats = emotion_processor.cache.get_stats()
cache_size = emotion_processor.cache.get_cache_size()

print(f"\n情感描述缓存:")
print(f"  命中次数: {cache_stats['description']['hits']}")
print(f"  未命中次数: {cache_stats['description']['misses']}")
print(f"  命中率: {cache_stats['description']['hit_rate']:.1f}%")

print(f"\n嵌入向量缓存:")
print(f"  命中次数: {cache_stats['embedding']['hits']}")
print(f"  未命中次数: {cache_stats['embedding']['misses']}")
print(f"  命中率: {cache_stats['embedding']['hit_rate']:.1f}%")

print(f"\n缓存文件数量:")
print(f"  描述缓存文件: {cache_size['description_cache_files']}")
print(f"  嵌入缓存文件: {cache_size['embedding_cache_files']}")
print(f"  总文件数: {cache_size['total_files']}")

print("\n" + "=" * 120)
print("【结果验证】")
print("=" * 120)

# 验证第二次运行的结果与第一次一致
all_match = True
differences = []

for i in range(min(len(nodes_1), len(nodes_2))):
    node1 = nodes_1[i]
    node2 = nodes_2[i]

    if node1.entity != node2.entity:
        all_match = False
        differences.append(f"实体不匹配: {node1.entity} vs {node2.entity}")
    elif not (node1.emotion_vector == node2.emotion_vector).all():
        all_match = False
        diff_norm = (node1.emotion_vector - node2.emotion_vector).norm()
        differences.append(f"向量差异: {node1.entity}, norm={diff_norm:.6f}")

if all_match:
    print("\n✓ 验证通过：缓存数据与原始数据完全一致")
else:
    print(f"\n⚠ 警告：发现 {len(differences)} 个差异")
    for diff in differences[:5]:
        print(f"  {diff}")

print("\n" + "=" * 120)
print("【总结】")
print("=" * 120)

if speedup_2 > 0:
    print(f"\n缓存加速效果（全缓存）: {speedup_2:.2f}x")
    print(f"时间节省: {time_first_run - time_second_run:.2f}s")

    if speedup_2 >= 10:
        print("\n✓✓✓ 缓存效果极其显著！达到10倍以上加速")
    elif speedup_2 >= 5:
        print("\n✓✓ 缓存效果显著！达到5-10倍加速")
    elif speedup_2 >= 2:
        print("\n✓ 缓存效果良好（2-5倍加速）")
    else:
        print("\n⚠ 缓存效果一般（2倍以下加速）")

print("\n下一步建议:")
print("  1. 如果验证通过 → 进行综合性能对比测试")
print("  2. 如果验证失败 → 检查缓存逻辑")

print("\n" + "=" * 120)
print("测试完成！")
print("=" * 120)
print(f"\n日志已保存到: {log_file}")
print(f"缓存目录: {cache_dir}")
