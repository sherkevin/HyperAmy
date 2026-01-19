#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
准备消融实验脚本

创建消融实验的测试脚本，包括：
1. 情绪模型vs LLM API对比
2. 检索方法消融（Amygdala vs HippoRAG等）
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 创建消融实验目录结构
ablation_dir = project_root / "test" / "ablation_experiments"
ablation_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("准备消融实验脚本")
print("=" * 80)
print(f"\n消融实验目录: {ablation_dir}")

# 实验1: 情绪模型vs LLM API对比
script1 = ablation_dir / "test_emotion_model_vs_llm_api.py"
if not script1.exists():
    content1 = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验1: 情绪模型vs LLM API对比

对比EmotionV2 (LLM API) vs EmotionV3 (emos模型)的性能
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from particle.emotion_v2 import EmotionV2
from particle.emotion_v3 import EmotionV3
import numpy as np

def test_emotion_model_comparison():
    """对比情绪模型和LLM API"""
    print("=" * 80)
    print("消融实验1: 情绪模型 vs LLM API")
    print("=" * 80)
    
    # 测试文本
    test_texts = [
        "I love this movie! It makes me feel happy and excited.",
        "This is terrible. I hate it so much.",
        "The weather is nice today. I feel calm and peaceful.",
    ] * 10  # 30个文本用于测试
    
    # 测试EmotionV2 (LLM API)
    print("\n1. 测试EmotionV2 (LLM API)...")
    emotion_v2 = EmotionV2(enable_cache=False)
    
    start_time = time.time()
    v2_results = []
    for text in test_texts:
        try:
            vector = emotion_v2.extract(text, validate=False)
            v2_results.append(vector)
        except Exception as e:
            print(f"  ⚠️ 错误: {e}")
            v2_results.append(None)
    v2_time = time.time() - start_time
    
    print(f"  完成时间: {v2_time:.2f}秒")
    print(f"  平均每个文本: {v2_time/len(test_texts):.2f}秒")
    print(f"  成功数: {len([r for r in v2_results if r is not None])}/{len(test_texts)}")
    
    # 测试EmotionV3 (emos模型)
    print("\n2. 测试EmotionV3 (emos模型)...")
    checkpoint_path = "/public/jiangh/emos/checkpoints/qwen3_8b/last_checkpoint.pt"
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    emotion_v3 = EmotionV3(
        emos_checkpoint_path=checkpoint_path,
        device=device,
        enable_cache=False,
    )
    
    start_time = time.time()
    v3_results = []
    for text in test_texts:
        try:
            nodes = emotion_v3.process(text=text, text_id="test", entities=None)
            if nodes:
                v3_results.append(nodes[0].emotion_vector)
            else:
                v3_results.append(None)
        except Exception as e:
            print(f"  ⚠️ 错误: {e}")
            v3_results.append(None)
    v3_time = time.time() - start_time
    
    print(f"  完成时间: {v3_time:.2f}秒")
    print(f"  平均每个文本: {v3_time/len(test_texts):.2f}秒")
    print(f"  成功数: {len([r for r in v3_results if r is not None])}/{len(test_texts)}")
    
    # 对比结果
    print("\n3. 性能对比:")
    print(f"  速度提升: {v2_time/v3_time:.2f}x" if v3_time > 0 else "  N/A")
    print(f"  时间节省: {(v2_time-v3_time)/v2_time*100:.1f}%" if v2_time > 0 else "  N/A")
    
    # 计算成本（粗略估算）
    # LLM API: 假设每个请求$0.001
    v2_cost = len([r for r in v2_results if r is not None]) * 0.001
    # emos模型: 本地运行，主要成本是GPU时间
    v3_cost = 0  # 本地运行，无API成本
    print(f"\n4. 成本对比:")
    print(f"  LLM API成本: ${v2_cost:.4f}")
    print(f"  emos模型成本: ${v3_cost:.4f} (本地运行)")
    print(f"  成本节省: ${v2_cost:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ 消融实验1完成")
    print("=" * 80)

if __name__ == "__main__":
    test_emotion_model_comparison()
'''
    script1.write_text(content1, encoding='utf-8')
    script1.chmod(0o755)
    print(f"✅ 已创建: {script1.name}")
else:
    print(f"⚠️ 已存在: {script1.name}")

print("\n✅ 消融实验脚本准备完成")
