#!/usr/bin/env python
"""测试 HotpotQA 数据集下载"""
import traceback
from datasets import load_dataset

try:
    print("尝试下载 HotpotQA 数据集...")
    # 尝试不同的下载模式
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    print(f"✅ 成功: {len(dataset)} 个样本")
except Exception as e:
    print(f"❌ 错误: {e}")
    traceback.print_exc()

