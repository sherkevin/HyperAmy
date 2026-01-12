#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试实体粒度数据集

验证：
1. 数据加载和格式验证
2. 实体位置匹配
3. soft_label维度验证
4. intensity计算验证
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_dataset(dataset_file: Path) -> Dict[str, Any]:
    """
    验证数据集格式和内容
    
    Args:
        dataset_file: 数据集文件路径
        
    Returns:
        验证结果字典
    """
    results = {
        'total_samples': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'total_targets': 0,
        'errors': []
    }
    
    if not dataset_file.exists():
        results['errors'].append(f"文件不存在: {dataset_file}")
        return results
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            
            results['total_samples'] += 1
            
            try:
                data = json.loads(line)
                
                # 验证必需字段
                if 'text' not in data:
                    results['errors'].append(f"行 {line_num}: 缺少 'text' 字段")
                    results['invalid_samples'] += 1
                    continue
                
                if 'targets' not in data:
                    results['errors'].append(f"行 {line_num}: 缺少 'targets' 字段")
                    results['invalid_samples'] += 1
                    continue
                
                text = data['text']
                targets = data['targets']
                
                if not isinstance(targets, list):
                    results['errors'].append(f"行 {line_num}: 'targets' 必须是列表")
                    results['invalid_samples'] += 1
                    continue
                
                # 验证每个target
                for target_idx, target in enumerate(targets):
                    results['total_targets'] += 1
                    
                    # 验证必需字段
                    required_fields = ['span_text', 'char_start', 'char_end', 'soft_label', 'intensity']
                    for field in required_fields:
                        if field not in target:
                            results['errors'].append(
                                f"行 {line_num}, target {target_idx}: 缺少 '{field}' 字段"
                            )
                            results['invalid_samples'] += 1
                            break
                    else:
                        # 验证字符位置
                        char_start = target['char_start']
                        char_end = target['char_end']
                        
                        if not isinstance(char_start, int) or not isinstance(char_end, int):
                            results['errors'].append(
                                f"行 {line_num}, target {target_idx}: char_start/char_end 必须是整数"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        if char_start >= char_end:
                            results['errors'].append(
                                f"行 {line_num}, target {target_idx}: char_start ({char_start}) >= char_end ({char_end})"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        # 验证span_text与文本位置匹配
                        span_text = target['span_text']
                        if char_end > len(text):
                            results['errors'].append(
                                f"行 {line_num}, target {target_idx}: char_end ({char_end}) > 文本长度 ({len(text)})"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        actual_span = text[char_start:char_end]
                        if actual_span.strip() != span_text.strip():
                            results['errors'].append(
                                f"行 {line_num}, target {target_idx}: span_text 不匹配 "
                                f"(期望: '{actual_span}', 实际: '{span_text}')"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        # 验证soft_label维度
                        soft_label = target['soft_label']
                        if not isinstance(soft_label, list):
                            results['errors'].append(
                                f"行 {line_num}, target {target_idx}: soft_label 必须是列表"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        if len(soft_label) != 28:
                            results['errors'].append(
                                f"行 {line_num}, target {target_idx}: soft_label 维度不是28 ({len(soft_label)})"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        # 验证soft_label是概率分布（和接近1.0）
                        soft_label_sum = sum(soft_label)
                        if abs(soft_label_sum - 1.0) > 0.1:  # 允许一定误差
                            results['errors'].append(
                                f"行 {line_num}, target {target_idx}: soft_label 和不为1.0 ({soft_label_sum:.4f})"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        # 验证intensity计算（max-norm）
                        intensity = target['intensity']
                        expected_intensity = max(soft_label)
                        if abs(intensity - expected_intensity) > 1e-6:
                            results['errors'].append(
                                f"行 {line_num}, target {target_idx}: intensity 计算错误 "
                                f"(期望: {expected_intensity:.6f}, 实际: {intensity:.6f})"
                            )
                            results['invalid_samples'] += 1
                            continue
                
                results['valid_samples'] += 1
                
            except json.JSONDecodeError as e:
                results['errors'].append(f"行 {line_num}: JSON解析失败: {e}")
                results['invalid_samples'] += 1
            except Exception as e:
                results['errors'].append(f"行 {line_num}: 验证失败: {e}")
                results['invalid_samples'] += 1
    
    return results


def print_validation_results(results: Dict[str, Any]):
    """打印验证结果"""
    print("=" * 80)
    print("数据集验证结果")
    print("=" * 80)
    print(f"总样本数: {results['total_samples']}")
    print(f"有效样本数: {results['valid_samples']}")
    print(f"无效样本数: {results['invalid_samples']}")
    print(f"总实体数: {results['total_targets']}")
    print()
    
    if results['errors']:
        print(f"错误数: {len(results['errors'])}")
        print("前10个错误:")
        for error in results['errors'][:10]:
            print(f"  - {error}")
        if len(results['errors']) > 10:
            print(f"  ... 还有 {len(results['errors']) - 10} 个错误")
    else:
        print("✅ 所有样本验证通过！")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试实体粒度数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/training/entity_granularity/entity_granularity_monte_cristo.jsonl",
        help="数据集文件路径"
    )
    
    args = parser.parse_args()
    
    dataset_file = Path(project_root) / args.dataset
    
    if not dataset_file.exists():
        print(f"❌ 数据集文件不存在: {dataset_file}")
        sys.exit(1)
    
    print(f"验证数据集: {dataset_file}")
    results = validate_dataset(dataset_file)
    print_validation_results(results)
    
    # 如果验证失败，退出码为1
    if results['invalid_samples'] > 0:
        sys.exit(1)

