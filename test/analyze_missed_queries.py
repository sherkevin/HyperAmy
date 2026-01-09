#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析未命中查询（40%完全未命中的查询）

这是任务2：未命中查询分析
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set
import logging
from collections import Counter

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experiment_results(result_file: Path) -> List[Dict[str, Any]]:
    """加载实验结果"""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def identify_missed_queries(results: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    识别未命中的查询
    
    Returns:
        {
            'all_missed': [索引列表],  # 三种方法都未命中
            'hipporag_only_missed': [索引列表],
            'fusion_only_missed': [索引列表],
            'hyperamy_only_missed': [索引列表],
            'hipporag_hit': [索引列表],
            'fusion_hit': [索引列表],
            'hyperamy_hit': [索引列表]
        }
    """
    all_missed = []
    hipporag_hit = []
    fusion_hit = []
    hyperamy_hit = []
    
    for idx, result in enumerate(results):
        hipporag_missed = not result.get('hipporag', {}).get('hit', False)
        fusion_missed = not result.get('fusion', {}).get('hit', False)
        hyperamy_missed = not result.get('hyperamy', {}).get('hit', False)
        
        if hipporag_missed and fusion_missed and hyperamy_missed:
            all_missed.append(idx)
        
        if not hipporag_missed:
            hipporag_hit.append(idx)
        if not fusion_missed:
            fusion_hit.append(idx)
        if not hyperamy_missed:
            hyperamy_hit.append(idx)
    
    # 计算只有某个方法命中的
    hipporag_only_missed = [i for i in range(len(results)) 
                            if i not in hipporag_hit and (i in fusion_hit or i in hyperamy_hit)]
    fusion_only_missed = [i for i in range(len(results)) 
                         if i not in fusion_hit and (i in hipporag_hit or i in hyperamy_hit)]
    hyperamy_only_missed = [i for i in range(len(results)) 
                           if i not in hyperamy_hit and (i in hipporag_hit or i in fusion_hit)]
    
    return {
        'all_missed': all_missed,
        'hipporag_only_missed': hipporag_only_missed,
        'fusion_only_missed': fusion_only_missed,
        'hyperamy_only_missed': hyperamy_only_missed,
        'hipporag_hit': hipporag_hit,
        'fusion_hit': fusion_hit,
        'hyperamy_hit': hyperamy_hit
    }


def analyze_query_characteristics(results: List[Dict[str, Any]], indices: List[int]) -> Dict[str, Any]:
    """分析查询特征"""
    if not indices:
        return {}
    
    # 加载QA数据
    qa_file = project_root / "data" / "public_benchmark" / "monte_cristo_qa_full.json"
    qa_data = []
    if qa_file.exists():
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    
    selected_queries = [qa_data[i] for i in indices if i < len(qa_data)]
    
    # 分析特征
    query_lengths = [len(q.get('question', '')) for q in selected_queries]
    answer_lengths = [len(q.get('answer', '')) for q in selected_queries]
    chunk_lengths = [len(q.get('chunk_text', '')) for q in selected_queries]
    
    # 关键词分析
    question_keywords = []
    for q in selected_queries:
        question = q.get('question', '').lower()
        # 提取可能的情绪/认知词汇
        emotion_words = ['恐惧', 'fear', '担心', 'worry', '直觉', 'instinct', '本能', 'instinctive',
                        '暗示', 'imply', '微妙', 'subtle', '情感', 'emotion', '情绪', 'sentiment']
        question_keywords.extend([w for w in emotion_words if w in question])
    
    keyword_counts = Counter(question_keywords)
    
    return {
        'count': len(indices),
        'avg_query_length': np.mean(query_lengths) if query_lengths else 0,
        'avg_answer_length': np.mean(answer_lengths) if answer_lengths else 0,
        'avg_chunk_length': np.mean(chunk_lengths) if chunk_lengths else 0,
        'top_keywords': dict(keyword_counts.most_common(10)),
        'requires_emotional_sensitivity': sum(1 for q in selected_queries 
                                             if q.get('requires_emotional_sensitivity', False))
    }


def analyze_gold_document_characteristics(results: List[Dict[str, Any]], indices: List[int]) -> Dict[str, Any]:
    """分析gold文档特征"""
    if not indices:
        return {}
    
    qa_file = project_root / "data" / "public_benchmark" / "monte_cristo_qa_full.json"
    qa_data = []
    if qa_file.exists():
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    
    selected_queries = [qa_data[i] for i in indices if i < len(qa_data)]
    
    chunk_lengths = [len(q.get('chunk_text', '')) for q in selected_queries]
    mass_values = [q.get('mass', 0.0) for q in selected_queries]
    
    return {
        'avg_chunk_length': np.mean(chunk_lengths) if chunk_lengths else 0,
        'avg_mass': np.mean(mass_values) if mass_values else 0,
        'min_mass': np.min(mass_values) if mass_values else 0,
        'max_mass': np.max(mass_values) if mass_values else 0
    }


def generate_analysis_report(results: List[Dict[str, Any]], missed_info: Dict[str, List[int]]) -> Dict[str, Any]:
    """生成分析报告"""
    report = {
        'summary': {
            'total_queries': len(results),
            'all_missed_count': len(missed_info['all_missed']),
            'all_missed_percentage': len(missed_info['all_missed']) / len(results) * 100,
            'hipporag_hit_count': len(missed_info['hipporag_hit']),
            'fusion_hit_count': len(missed_info['fusion_hit']),
            'hyperamy_hit_count': len(missed_info['hyperamy_hit'])
        },
        'all_missed_queries': {
            'indices': missed_info['all_missed'],
            'count': len(missed_info['all_missed']),
            'query_characteristics': analyze_query_characteristics(results, missed_info['all_missed']),
            'gold_document_characteristics': analyze_gold_document_characteristics(results, missed_info['all_missed'])
        },
        'method_specific_analysis': {
            'hipporag_only_missed': {
                'indices': missed_info['hipporag_only_missed'],
                'count': len(missed_info['hipporag_only_missed']),
                'characteristics': analyze_query_characteristics(results, missed_info['hipporag_only_missed'])
            },
            'fusion_only_missed': {
                'indices': missed_info['fusion_only_missed'],
                'count': len(missed_info['fusion_only_missed']),
                'characteristics': analyze_query_characteristics(results, missed_info['fusion_only_missed'])
            },
            'hyperamy_only_missed': {
                'indices': missed_info['hyperamy_only_missed'],
                'count': len(missed_info['hyperamy_only_missed']),
                'characteristics': analyze_query_characteristics(results, missed_info['hyperamy_only_missed'])
            }
        }
    }
    
    return report


def main():
    """主函数"""
    print("=" * 80)
    print("未命中查询深度分析")
    print("=" * 80)
    
    # 加载实验结果
    result_file = project_root / "outputs" / "three_methods_comparison_monte_cristo" / "comparison_results.json"
    if not result_file.exists():
        logger.error(f"实验结果文件不存在: {result_file}")
        return
    
    logger.info(f"加载实验结果: {result_file}")
    results = load_experiment_results(result_file)
    logger.info(f"共 {len(results)} 个查询结果")
    
    # 识别未命中查询
    missed_info = identify_missed_queries(results)
    
    print(f"\n【总体统计】")
    print(f"总查询数: {len(results)}")
    print(f"完全未命中（三种方法都未命中）: {len(missed_info['all_missed'])} ({len(missed_info['all_missed'])/len(results)*100:.1f}%)")
    print(f"HippoRAG命中: {len(missed_info['hipporag_hit'])} ({len(missed_info['hipporag_hit'])/len(results)*100:.1f}%)")
    print(f"Fusion命中: {len(missed_info['fusion_hit'])} ({len(missed_info['fusion_hit'])/len(results)*100:.1f}%)")
    print(f"HyperAmy命中: {len(missed_info['hyperamy_hit'])} ({len(missed_info['hyperamy_hit'])/len(results)*100:.1f}%)")
    
    # 生成详细分析报告
    report = generate_analysis_report(results, missed_info)
    
    # 打印详细分析
    print(f"\n【完全未命中查询分析】")
    print(f"数量: {report['all_missed_queries']['count']}")
    print(f"\n查询特征:")
    qc = report['all_missed_queries']['query_characteristics']
    print(f"  平均查询长度: {qc.get('avg_query_length', 0):.1f} 字符")
    print(f"  平均答案长度: {qc.get('avg_answer_length', 0):.1f} 字符")
    print(f"  平均chunk长度: {qc.get('avg_chunk_length', 0):.1f} 字符")
    print(f"  需要情感敏感性: {qc.get('requires_emotional_sensitivity', 0)}/{qc.get('count', 0)}")
    if qc.get('top_keywords'):
        print(f"  关键词: {qc.get('top_keywords', {})}")
    
    print(f"\nGold文档特征:")
    gdc = report['all_missed_queries']['gold_document_characteristics']
    print(f"  平均chunk长度: {gdc.get('avg_chunk_length', 0):.1f} 字符")
    print(f"  平均mass值: {gdc.get('avg_mass', 0):.3f}")
    print(f"  mass范围: [{gdc.get('min_mass', 0):.3f}, {gdc.get('max_mass', 0):.3f}]")
    
    # 保存报告
    output_file = project_root / "outputs" / "three_methods_comparison_monte_cristo" / "missed_queries_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ 分析报告已保存到: {output_file}")
    
    # 打印未命中查询的详细信息（前10个）
    print(f"\n【未命中查询详细信息（前10个）】")
    qa_file = project_root / "data" / "public_benchmark" / "monte_cristo_qa_full.json"
    if qa_file.exists():
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        for i, idx in enumerate(missed_info['all_missed'][:10]):
            if idx < len(qa_data):
                qa = qa_data[idx]
                print(f"\n查询 {idx+1}:")
                print(f"  问题: {qa.get('question', '')[:100]}...")
                print(f"  Chunk ID: {qa.get('chunk_id', '')}")
                print(f"  Chunk Text: {qa.get('chunk_text', '')[:150]}...")
    
    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

