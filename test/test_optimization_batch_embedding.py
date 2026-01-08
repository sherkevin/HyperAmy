#!/usr/bin/env python3
"""
优化方案二测试：批量Embedding调用

测试目标：
1. 验证批量Embedding实现的正确性
2. 测量性能提升倍数
3. 确保生成的嵌入向量质量不变
"""

import logging
import sys
import time
from pathlib import Path

# 设置日志
log_file = Path("./log/test_optimization_batch_embedding.log")
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
import numpy as np

print("=" * 120)
print("优化方案二测试：批量Embedding调用")
print("=" * 120)

# 测试数据
test_sentence = '''Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate.
"You refuse?" she whispered, her voice full of tears. "Is it because you are our enemy?
To refuse to break bread... means you bring death to this house.'''

test_entities = ["Mercedes", "terror", "eyes", "hand", "plate", "refuse", "tears", "enemy", "bread", "death"]

test_descriptions = {
    "Mercedes": "terror, fear, sadness, despair, anxiety, sorrow",
    "terror": "fear, dread, despair, anguish, anxiety, distress",
    "eyes": "terror, fear, sadness, distress, anxiety, despair",
    "hand": "fear, anxiety, distress, helplessness, dread, agitation",
    "plate": "fear, dread, anxiety, distress, despair",
    "refuse": "fear, terror, sadness, despair, betrayal, anxiety",
    "tears": "fear, despair, betrayal, sorrow, anguish",
    "enemy": "fear, hostility, sadness, dread, distrust, sorrow",
    "bread": "terror, fear, despair, betrayal, dread",
    "death": "fear, terror, dread, despair, anguish"
}

print(f"\n测试实体数量: {len(test_entities)}")
print(f"测试实体列表: {test_entities}")

# 初始化
print("\n" + "=" * 120)
print("【初始化】")
print("=" * 120)

emotion_processor = EmotionV2()
print("✓ EmotionV2处理器初始化完成")

# ========== 测试1：原始串行Embedding版本 ==========
print("\n" + "=" * 120)
print("【测试 1】原始串行Embedding版本（baseline）")
print("=" * 120)

start_time = time.time()
embeddings_serial = []
for entity in test_entities:
    description = test_descriptions[entity]
    embedding = emotion_processor._get_emotion_embedding(description)
    embeddings_serial.append(embedding)
serial_time = time.time() - start_time

print(f"\n✓ 串行版本完成 (耗时: {serial_time:.2f}s)")
print(f"成功生成: {len([e for e in embeddings_serial if e.size > 0])}/{len(test_entities)} 个向量")

# 显示部分结果
for i, entity in enumerate(test_entities[:3]):
    embedding = embeddings_serial[i]
    print(f"  {entity}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.6f}")

# ========== 测试2：批量Embedding版本（方案二） ==========
print("\n" + "=" * 120)
print("【测试 2】批量Embedding优化版本（方案二）")
print("=" * 120)

start_time = time.time()
descriptions_list = [test_descriptions[entity] for entity in test_entities]
embeddings_batch = emotion_processor._batch_get_emotion_embeddings(descriptions_list)
batch_time = time.time() - start_time

print(f"\n✓ 批量版本完成 (耗时: {batch_time:.2f}s)")
print(f"成功生成: {len([e for e in embeddings_batch if e.size > 0])}/{len(test_entities)} 个向量")

# 显示部分结果
for i, entity in enumerate(test_entities[:3]):
    embedding = embeddings_batch[i]
    print(f"  {entity}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.6f}")

# ========== 对比分析 ==========
print("\n" + "=" * 120)
print("【性能对比分析】")
print("=" * 120)

print(f"\n{'版本':<20} {'耗时(s)':<15} {'提升倍数':<15} {'成功率':<15}")
print("-" * 75)

serial_success = len([e for e in embeddings_serial if e.size > 0]) / len(test_entities) * 100
batch_success = len([e for e in embeddings_batch if e.size > 0]) / len(test_entities) * 100
speedup = serial_time / batch_time if batch_time > 0 else 0

print(f"{'串行版本':<20} {serial_time:<15.2f} {'1.0x':<15} {serial_success:.1f}%")
print(f"{'批量版本':<20} {batch_time:<15.2f} {speedup:<15.2f}x {batch_success:.1f}%")

print("\n" + "=" * 120)
print("【结果验证】")
print("=" * 120)

# 验证生成的嵌入向量是否一致
all_match = True
differences = []

for i, entity in enumerate(test_entities):
    emb_serial = embeddings_serial[i]
    emb_batch = embeddings_batch[i]

    # 比较向量
    if emb_serial.shape != emb_batch.shape:
        all_match = False
        differences.append({
            'entity': entity,
            'reason': f'shape mismatch: {emb_serial.shape} vs {emb_batch.shape}'
        })
    elif not np.allclose(emb_serial, emb_batch, rtol=1e-5):
        all_match = False
        diff_norm = np.linalg.norm(emb_serial - emb_batch)
        differences.append({
            'entity': entity,
            'reason': f'vector difference: norm={diff_norm:.6f}'
        })

if all_match:
    print("\n✓ 验证通过：批量版本生成的嵌入向量与串行版本完全一致")
else:
    print(f"\n⚠ 警告：发现 {len(differences)} 个实体的嵌入向量存在差异")
    print("\n差异详情:")
    for diff in differences[:5]:
        print(f"  {diff['entity']}: {diff['reason']}")

    # 检查差异是否可接受（由于浮点数精度）
    max_diff = max([np.linalg.norm(embeddings_serial[i] - embeddings_batch[i])
                    for i in range(len(test_entities))
                    if embeddings_serial[i].shape == embeddings_batch[i].shape])
    if max_diff < 1e-4:
        print(f"\n✓ 差异在可接受范围内（最大差异: {max_diff:.8f}），可能为浮点数精度问题")
    else:
        print(f"\n⚠ 差异较大（最大差异: {max_diff:.6f}），需要检查")

print("\n" + "=" * 120)
print("【总结】")
print("=" * 120)

if speedup > 0:
    print(f"\n性能提升: {speedup:.2f}x (从 {serial_time:.2f}s 降至 {batch_time:.2f}s)")
    print(f"时间节省: {serial_time - batch_time:.2f}s")

    if speedup >= 5:
        print("\n✓✓✓ 优化效果显著！达到预期目标（5倍以上提升）")
    elif speedup >= 2:
        print("\n✓✓ 优化效果良好（2-5倍提升）")
    elif speedup >= 1.2:
        print("\n✓ 优化效果一般（1.2-2倍提升）")
    else:
        print("\n⚠ 优化效果不明显，可能API已支持批量处理")
else:
    print("\n批量版本耗时接近或低于串行版本，说明优化有效")

print("\n下一步建议:")
print("  1. 如果验证通过 → 进入方案三：缓存机制")
print("  2. 如果向量不一致 → 检查批量API调用逻辑")

print("\n" + "=" * 120)
print("测试完成！")
print("=" * 120)
print(f"\n日志已保存到: {log_file}")
