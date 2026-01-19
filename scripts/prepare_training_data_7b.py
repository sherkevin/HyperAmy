#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为7B模型准备训练数据

将entity_granularity_v2_full.jsonl分割为训练集和验证集
"""
import json
import random
from pathlib import Path
from typing import List, Dict

def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: Path):
    """保存为JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 数据路径
    input_file = Path("data/training/entity_granularity/entity_granularity_v2_full.jsonl")
    train_file = Path("emos-master/data/train_7b.jsonl")
    val_file = Path("emos-master/data/val_7b.jsonl")
    
    # 创建输出目录
    train_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("准备7B模型训练数据")
    print("=" * 70)
    
    # 加载数据
    print(f"\n加载数据: {input_file}")
    data = load_jsonl(input_file)
    print(f"总样本数: {len(data)}")
    
    # 展平数据（每个target成为一个训练样本）
    flattened_data = []
    for item in data:
        text = item.get('text', '')
        targets = item.get('targets', [])
        for target in targets:
            flattened_data.append({
                'text': text,
                'char_start': target.get('char_start'),
                'char_end': target.get('char_end'),
                'span_text': target.get('span_text', ''),
                'soft_label': target.get('soft_label', {}),
            })
    
    print(f"展平后样本数: {len(flattened_data)}")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(flattened_data)
    
    # 分割（80%训练，20%验证）
    split_idx = int(len(flattened_data) * 0.8)
    train_data = flattened_data[:split_idx]
    val_data = flattened_data[split_idx:]
    
    # 保存
    print(f"\n保存训练集: {train_file} ({len(train_data)} 样本)")
    save_jsonl(train_data, train_file)
    
    print(f"保存验证集: {val_file} ({len(val_data)} 样本)")
    save_jsonl(val_data, val_file)
    
    print("\n✅ 数据准备完成！")
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    print(f"总计: {len(flattened_data)} 样本")

if __name__ == "__main__":
    main()
