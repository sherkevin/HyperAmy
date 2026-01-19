#!/usr/bin/env python
"""测试 HotpotQA 数据集下载 - 使用不同方法"""
import traceback
from datasets import load_dataset

try:
    print("方法1: 使用 trust_remote_code...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    print(f"✅ 成功: {len(dataset)} 个样本")
except Exception as e1:
    print(f"❌ 方法1失败: {e1}")
    try:
        print("\n方法2: 使用 revision...")
        dataset = load_dataset("hotpot_qa", "distractor", split="validation", revision="main")
        print(f"✅ 成功: {len(dataset)} 个样本")
    except Exception as e2:
        print(f"❌ 方法2失败: {e2}")
        try:
            print("\n方法3: 直接从 HuggingFace Hub 下载...")
            from huggingface_hub import hf_hub_download
            import json
            # 尝试直接下载文件
            print("   尝试下载原始文件...")
            # 这个方法需要知道具体的文件路径
            print("   需要手动指定文件路径")
        except Exception as e3:
            print(f"❌ 方法3失败: {e3}")
            traceback.print_exc()

