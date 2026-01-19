#!/usr/bin/env python3
"""
创建训练数据的子集用于不同阶段的测试。

支持创建：
- 小规模测试集（50-100个样本）
- 中规模验证集（300-500个样本）
- 自定义大小子集

使用方法:
    python scripts/create_test_subsets.py \
        --input_train emos-master/data/train.jsonl \
        --input_val emos-master/data/val.jsonl \
        --train_samples 50 \
        --val_samples 10 \
        --output_dir emos-master/data
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def sample_jsonl(input_file: Path, num_samples: int, output_file: Path):
    """
    从JSONL文件中采样指定数量的样本。
    
    Args:
        input_file: 输入JSONL文件路径
        num_samples: 需要采样的样本数
        output_file: 输出JSONL文件路径
    """
    samples: List[Dict[str, Any]] = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                    if len(samples) >= num_samples:
                        break
                except json.JSONDecodeError as e:
                    print(f"警告: 跳过无效JSON行: {e}")
                    continue
    
    # 写入输出文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 统计展开后的样本数（实体数）
    total_entities = sum(len(sample.get('targets', [])) for sample in samples)
    
    print(f"✓ 从 {input_file} 采样了 {len(samples)} 个原始样本")
    print(f"  展开后样本数（实体数）: {total_entities}")
    print(f"  已保存到: {output_file}")
    
    return len(samples), total_entities


def main():
    parser = argparse.ArgumentParser(
        description="创建训练数据的子集用于测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 创建小规模测试集（50个训练样本，10个验证样本）
  python scripts/create_test_subsets.py \\
      --input_train emos-master/data/train.jsonl \\
      --input_val emos-master/data/val.jsonl \\
      --train_samples 50 \\
      --val_samples 10 \\
      --output_dir emos-master/data \\
      --suffix small

  # 创建中规模验证集（400个训练样本，50个验证样本）
  python scripts/create_test_subsets.py \\
      --input_train emos-master/data/train.jsonl \\
      --input_val emos-master/data/val.jsonl \\
      --train_samples 400 \\
      --val_samples 50 \\
      --output_dir emos-master/data \\
      --suffix medium
        """
    )
    
    parser.add_argument(
        "--input_train",
        type=str,
        required=True,
        help="完整训练集JSONL文件路径"
    )
    
    parser.add_argument(
        "--input_val",
        type=str,
        required=True,
        help="完整验证集JSONL文件路径"
    )
    
    parser.add_argument(
        "--train_samples",
        type=int,
        required=True,
        help="采样的训练样本数"
    )
    
    parser.add_argument(
        "--val_samples",
        type=int,
        required=True,
        help="采样的验证样本数"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="emos-master/data",
        help="输出目录"
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        default="subset",
        help="输出文件名后缀（将生成 train_<suffix>.jsonl 和 val_<suffix>.jsonl）"
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    input_train = Path(args.input_train)
    input_val = Path(args.input_val)
    output_dir = Path(args.output_dir)
    
    # 检查输入文件是否存在
    if not input_train.exists():
        print(f"错误: 训练集文件不存在: {input_train}")
        return 1
    
    if not input_val.exists():
        print(f"错误: 验证集文件不存在: {input_val}")
        return 1
    
    # 确定输出文件路径
    train_output = output_dir / f"train_{args.suffix}.jsonl"
    val_output = output_dir / f"val_{args.suffix}.jsonl"
    
    print("=" * 60)
    print("创建训练数据子集")
    print("=" * 60)
    print(f"输入训练集: {input_train}")
    print(f"输入验证集: {input_val}")
    print(f"训练样本数: {args.train_samples}")
    print(f"验证样本数: {args.val_samples}")
    print(f"输出目录: {output_dir}")
    print(f"文件后缀: {args.suffix}")
    print("=" * 60)
    print()
    
    # 采样训练集
    print("【步骤1】采样训练集...")
    train_count, train_entities = sample_jsonl(
        input_train,
        args.train_samples,
        train_output
    )
    print()
    
    # 采样验证集
    print("【步骤2】采样验证集...")
    val_count, val_entities = sample_jsonl(
        input_val,
        args.val_samples,
        val_output
    )
    print()
    
    # 总结
    print("=" * 60)
    print("采样完成！")
    print("=" * 60)
    print(f"训练集: {train_count} 个原始样本 → {train_entities} 个展开样本")
    print(f"验证集: {val_count} 个原始样本 → {val_entities} 个展开样本")
    print(f"输出文件:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
