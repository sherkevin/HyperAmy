#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将HyperAmy的entity_granularity_v2数据集转换为emos训练格式

格式转换：
- HyperAmy格式: {"text": "...", "targets": [{"span_text": "...", "char_start": ..., "char_end": ..., "soft_label": [...], "intensity": ...}]}
- emos格式: {"text": "...", "targets": [{"span_text": "...", "char_start": ..., "char_end": ..., "soft_label": [...]}]}

差异：emos不需要intensity字段
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def convert_sample(hyperamy_sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    转换单个样本从HyperAmy格式到emos格式
    
    Args:
        hyperamy_sample: HyperAmy格式的样本
        
    Returns:
        emos格式的样本
    """
    emos_sample = {
        "text": hyperamy_sample["text"],
        "targets": []
    }
    
    for target in hyperamy_sample.get("targets", []):
        emos_target = {
            "span_text": target["span_text"],
            "char_start": target["char_start"],
            "char_end": target["char_end"],
            "soft_label": target["soft_label"]  # 28维列表
        }
        # emos不需要intensity字段，所以不包含
        
        emos_sample["targets"].append(emos_target)
    
    return emos_sample


def convert_dataset(
    input_file: str,
    output_file: str,
    max_samples: int = None,
    train_ratio: float = 0.9
):
    """
    转换整个数据集
    
    Args:
        input_file: HyperAmy格式的输入JSONL文件
        output_file: 输出目录（会生成train.jsonl和val.jsonl）
        max_samples: 最大样本数（None表示使用全部）
        train_ratio: 训练集比例
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 读取所有样本
    samples = []
    print(f"读取数据: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    sample = json.loads(line)
                    emos_sample = convert_sample(sample)
                    samples.append(emos_sample)
                except json.JSONDecodeError as e:
                    print(f"⚠️  第{line_num}行JSON解析错误: {e}")
                    continue
    
    print(f"✅ 读取了 {len(samples)} 个样本")
    
    # 限制样本数
    if max_samples and len(samples) > max_samples:
        print(f"限制样本数: {len(samples)} -> {max_samples}")
        samples = samples[:max_samples]
    
    # 分割训练集和验证集
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"训练集: {len(train_samples)} 个样本")
    print(f"验证集: {len(val_samples)} 个样本")
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入训练集
    train_file = output_path.parent / "train.jsonl"
    print(f"写入训练集: {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 写入验证集
    val_file = output_path.parent / "val.jsonl"
    print(f"写入验证集: {val_file}")
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print("✅ 转换完成！")
    print(f"训练集: {train_file}")
    print(f"验证集: {val_file}")
    
    # 验证格式
    print("\n验证格式...")
    with open(train_file, 'r') as f:
        test_sample = json.loads(f.readline())
        if "text" in test_sample and "targets" in test_sample:
            print("✅ 格式验证通过")
            print(f"   示例文本: {test_sample['text'][:50]}...")
            print(f"   目标数量: {len(test_sample['targets'])}")
            if test_sample['targets']:
                target = test_sample['targets'][0]
                print(f"   soft_label维度: {len(target.get('soft_label', []))}")
        else:
            print("❌ 格式验证失败")


def main():
    parser = argparse.ArgumentParser(description="转换HyperAmy数据集到emos格式")
    parser.add_argument(
        "--input",
        type=str,
        default="data/training/entity_granularity/entity_granularity_v2_full.jsonl",
        help="HyperAmy格式的输入文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="emos-master/data/hyperamy_train.jsonl",
        help="输出文件路径（会生成train.jsonl和val.jsonl在同一目录）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数（None表示使用全部）"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练集比例（默认0.9）"
    )
    
    args = parser.parse_args()
    
    convert_dataset(
        args.input,
        args.output,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio
    )


if __name__ == "__main__":
    main()
