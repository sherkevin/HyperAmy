#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证实体粒度数据集v2的质量

检查项：
1. 数据格式正确性
2. soft_label不归一化（和不一定=1.0）
3. intensity使用L2-norm
4. 章节标题过滤（不应包含大量Chapter）
5. 文本质量（长度、内容）
6. 实体质量（位置匹配、数量）
7. QA对数据（如果包含）
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def is_chapter_title(text: str) -> bool:
    """判断是否是章节标题"""
    text_clean = text.strip()
    if len(text_clean) < 50 and re.search(r'\bChapter\s+\d+', text_clean, re.IGNORECASE):
        return True
    lines = text_clean.split('\n')
    if len(lines) <= 2:
        for line in lines:
            if re.search(r'\bChapter\s+\d+', line, re.IGNORECASE):
                return True
    return False


def validate_dataset_v2(dataset_file: Path) -> Dict[str, Any]:
    """
    验证数据集v2的质量
    
    Returns:
        验证结果字典
    """
    results = {
        'total_samples': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'total_entities': 0,
        'errors': [],
        'warnings': [],
        'statistics': {
            'text_lengths': [],
            'entities_per_sample': [],
            'intensities': [],
            'soft_label_sums': [],
            'chapter_title_count': 0,
            'avg_entities_per_sample': 0,
            'avg_text_length': 0,
            'avg_intensity': 0,
        }
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
                
                # 检查文本质量
                results['statistics']['text_lengths'].append(len(text))
                
                # 检查是否是章节标题（应该被过滤）
                if is_chapter_title(text):
                    results['statistics']['chapter_title_count'] += 1
                    results['warnings'].append(f"行 {line_num}: 包含章节标题（应该被过滤）")
                
                # 验证每个target
                sample_entities = 0
                for target_idx, target in enumerate(targets):
                    results['total_entities'] += 1
                    sample_entities += 1
                    
                    # 验证必需字段
                    required_fields = ['span_text', 'char_start', 'char_end', 'soft_label', 'intensity']
                    for field in required_fields:
                        if field not in target:
                            results['errors'].append(
                                f"行 {line_num}, 实体 {target_idx}: 缺少 '{field}' 字段"
                            )
                            results['invalid_samples'] += 1
                            break
                    else:
                        # 验证字符位置
                        char_start = target['char_start']
                        char_end = target['char_end']
                        
                        if not isinstance(char_start, int) or not isinstance(char_end, int):
                            results['errors'].append(
                                f"行 {line_num}, 实体 {target_idx}: char_start/char_end 必须是整数"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        if char_start >= char_end:
                            results['errors'].append(
                                f"行 {line_num}, 实体 {target_idx}: char_start ({char_start}) >= char_end ({char_end})"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        # 验证span_text与文本位置匹配
                        span_text = target['span_text']
                        if char_end > len(text):
                            results['errors'].append(
                                f"行 {line_num}, 实体 {target_idx}: char_end ({char_end}) > 文本长度 ({len(text)})"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        actual_span = text[char_start:char_end]
                        if actual_span.strip() != span_text.strip():
                            results['warnings'].append(
                                f"行 {line_num}, 实体 {target_idx}: span_text 不完全匹配 "
                                f"(期望: '{actual_span[:30]}', 实际: '{span_text[:30]}')"
                            )
                        
                        # 验证soft_label维度
                        soft_label = target['soft_label']
                        if not isinstance(soft_label, list):
                            results['errors'].append(
                                f"行 {line_num}, 实体 {target_idx}: soft_label 必须是列表"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        if len(soft_label) != 28:
                            results['errors'].append(
                                f"行 {line_num}, 实体 {target_idx}: soft_label 维度不是28 ({len(soft_label)})"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        # 验证soft_label范围（应该在[0, 1]）
                        if soft_label:
                            min_val = min(soft_label)
                            max_val = max(soft_label)
                            if min_val < 0 or max_val > 1:
                                results['errors'].append(
                                    f"行 {line_num}, 实体 {target_idx}: soft_label 超出[0,1]范围 "
                                    f"([{min_val:.4f}, {max_val:.4f}])"
                                )
                                results['invalid_samples'] += 1
                                continue
                            
                            # 记录soft_label的和（v2不归一化，和不一定=1.0）
                            soft_label_sum = sum(soft_label)
                            results['statistics']['soft_label_sums'].append(soft_label_sum)
                            
                            # 如果和接近1.0，可能是旧格式，给出警告
                            if abs(soft_label_sum - 1.0) < 0.01:
                                results['warnings'].append(
                                    f"行 {line_num}, 实体 {target_idx}: soft_label 和接近1.0 ({soft_label_sum:.4f})，"
                                    f"可能是旧格式（v2不归一化）"
                                )
                        
                        # 验证intensity计算（v2使用L2-norm）
                        intensity = target['intensity']
                        if intensity < 0:
                            results['errors'].append(
                                f"行 {line_num}, 实体 {target_idx}: intensity 无效 ({intensity})"
                            )
                            results['invalid_samples'] += 1
                            continue
                        
                        # 检查intensity是否等于L2-norm（v2格式）
                        if soft_label and len(soft_label) == 28:
                            expected_intensity = np.linalg.norm(soft_label)
                            if abs(intensity - expected_intensity) > 0.001:
                                results['errors'].append(
                                    f"行 {line_num}, 实体 {target_idx}: intensity 计算错误 "
                                    f"(期望L2-norm: {expected_intensity:.6f}, 实际: {intensity:.6f})"
                                )
                                results['invalid_samples'] += 1
                                continue
                            
                            results['statistics']['intensities'].append(intensity)
                
                results['statistics']['entities_per_sample'].append(sample_entities)
                results['valid_samples'] += 1
                
            except json.JSONDecodeError as e:
                results['errors'].append(f"行 {line_num}: JSON解析失败: {e}")
                results['invalid_samples'] += 1
            except Exception as e:
                results['errors'].append(f"行 {line_num}: 验证失败: {e}")
                results['invalid_samples'] += 1
    
    # 计算统计信息
    if results['statistics']['text_lengths']:
        results['statistics']['avg_text_length'] = sum(results['statistics']['text_lengths']) / len(results['statistics']['text_lengths'])
    if results['statistics']['entities_per_sample']:
        results['statistics']['avg_entities_per_sample'] = sum(results['statistics']['entities_per_sample']) / len(results['statistics']['entities_per_sample'])
    if results['statistics']['intensities']:
        results['statistics']['avg_intensity'] = sum(results['statistics']['intensities']) / len(results['statistics']['intensities'])
    
    return results


def print_validation_results(results: Dict[str, Any]):
    """打印验证结果"""
    print("=" * 80)
    print("实体粒度数据集v2质量验证结果")
    print("=" * 80)
    print()
    print(f"总样本数: {results['total_samples']}")
    print(f"有效样本数: {results['valid_samples']}")
    print(f"无效样本数: {results['invalid_samples']}")
    print(f"总实体数: {results['total_entities']}")
    print()
    
    # 统计信息
    stats = results['statistics']
    print("📊 统计信息：")
    if stats['text_lengths']:
        print(f"  平均文本长度: {stats['avg_text_length']:.1f} 字符")
        print(f"  文本长度范围: [{min(stats['text_lengths'])}, {max(stats['text_lengths'])}]")
    if stats['entities_per_sample']:
        print(f"  平均实体数/样本: {stats['avg_entities_per_sample']:.2f}")
        print(f"  实体数范围: [{min(stats['entities_per_sample'])}, {max(stats['entities_per_sample'])}]")
    if stats['intensities']:
        print(f"  平均intensity: {stats['avg_intensity']:.4f}")
        print(f"  intensity范围: [{min(stats['intensities']):.4f}, {max(stats['intensities']):.4f}]")
    if stats['soft_label_sums']:
        avg_sum = sum(stats['soft_label_sums']) / len(stats['soft_label_sums'])
        min_sum = min(stats['soft_label_sums'])
        max_sum = max(stats['soft_label_sums'])
        print(f"  soft_label和（平均）: {avg_sum:.4f} (范围: [{min_sum:.4f}, {max_sum:.4f}])")
        print(f"  说明: v2格式不归一化，和不一定=1.0 ✅")
    print(f"  章节标题数量: {stats['chapter_title_count']} (应该接近0) {'✅' if stats['chapter_title_count'] < results['total_samples'] * 0.01 else '⚠️'}")
    print()
    
    # 错误
    if results['errors']:
        print(f"❌ 错误数: {len(results['errors'])}")
        print("前10个错误:")
        for error in results['errors'][:10]:
            print(f"  - {error}")
        if len(results['errors']) > 10:
            print(f"  ... 还有 {len(results['errors']) - 10} 个错误")
        print()
    else:
        print("✅ 无错误")
        print()
    
    # 警告
    if results['warnings']:
        print(f"⚠️  警告数: {len(results['warnings'])}")
        print("前10个警告:")
        for warning in results['warnings'][:10]:
            print(f"  - {warning}")
        if len(results['warnings']) > 10:
            print(f"  ... 还有 {len(results['warnings']) - 10} 个警告")
        print()
    else:
        print("✅ 无警告")
        print()
    
    # 质量评估
    print("🎯 质量评估：")
    quality_score = 0
    max_score = 5
    
    # 1. 格式正确性
    if results['invalid_samples'] == 0:
        print("  ✅ 格式正确性: 通过")
        quality_score += 1
    else:
        error_rate = results['invalid_samples'] / results['total_samples']
        if error_rate < 0.01:
            print(f"  ⚠️  格式正确性: 有少量错误 ({results['invalid_samples']}/{results['total_samples']})")
            quality_score += 0.5
        else:
            print(f"  ❌ 格式正确性: 错误率过高 ({error_rate:.2%})")
    
    # 2. soft_label不归一化
    if stats['soft_label_sums']:
        avg_sum = sum(stats['soft_label_sums']) / len(stats['soft_label_sums'])
        if abs(avg_sum - 1.0) > 0.1:  # 和明显不等于1.0
            print(f"  ✅ soft_label不归一化: 通过 (平均和={avg_sum:.4f})")
            quality_score += 1
        else:
            print(f"  ⚠️  soft_label不归一化: 平均和接近1.0 ({avg_sum:.4f})，可能是旧格式")
    
    # 3. intensity使用L2-norm
    if results['errors']:
        l2_norm_errors = sum(1 for e in results['errors'] if 'intensity' in e.lower() and 'L2-norm' in e.lower())
        if l2_norm_errors == 0:
            print("  ✅ intensity使用L2-norm: 通过")
            quality_score += 1
        else:
            print(f"  ❌ intensity使用L2-norm: 有 {l2_norm_errors} 个错误")
    else:
        print("  ✅ intensity使用L2-norm: 通过")
        quality_score += 1
    
    # 4. 章节标题过滤
    chapter_rate = stats['chapter_title_count'] / results['total_samples'] if results['total_samples'] > 0 else 0
    if chapter_rate < 0.01:
        print(f"  ✅ 章节标题过滤: 通过 (过滤率={chapter_rate:.2%})")
        quality_score += 1
    else:
        print(f"  ⚠️  章节标题过滤: 仍有 {stats['chapter_title_count']} 个章节标题 ({chapter_rate:.2%})")
        quality_score += 0.5
    
    # 5. 数据量
    if results['total_samples'] >= 100:
        print(f"  ✅ 数据量: 充足 ({results['total_samples']} 样本)")
        quality_score += 1
    elif results['total_samples'] >= 50:
        print(f"  ⚠️  数据量: 中等 ({results['total_samples']} 样本)")
        quality_score += 0.5
    else:
        print(f"  ❌ 数据量: 不足 ({results['total_samples']} 样本)")
    
    print()
    print(f"质量得分: {quality_score}/{max_score} ({quality_score/max_score*100:.0f}%)")
    print("=" * 80)
    
    return quality_score >= 4.0  # 质量得分>=4.0认为合格


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证实体粒度数据集v2质量")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/training/entity_granularity/entity_granularity_v2_full.jsonl",
        help="数据集文件路径"
    )
    
    args = parser.parse_args()
    
    dataset_file = Path(project_root) / args.dataset
    
    if not dataset_file.exists():
        print(f"❌ 数据集文件不存在: {dataset_file}")
        sys.exit(1)
    
    print(f"验证数据集: {dataset_file}")
    results = validate_dataset_v2(dataset_file)
    is_valid = print_validation_results(results)
    
    # 如果验证失败，退出码为1
    sys.exit(0 if is_valid else 1)
