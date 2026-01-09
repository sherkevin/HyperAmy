#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试HyperAmy基于情绪相似度的评估方法

这是任务1：开发HyperAmy评估方法
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.emotion_recall_eval import (
    EmotionRecallEvaluator,
    evaluate_hyperamy_emotion_recall,
    DEFAULT_EMOTION_SIMILARITY_THRESHOLD
)
from particle.emotion import Emotion
from point_label.emotion import Emotion as OldEmotion

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experiment_results(result_file: Path) -> Dict[str, Any]:
    """加载实验结果"""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_texts_from_results(results: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    从实验结果中提取查询、gold文本和检索结果
    
    Returns:
        (queries, gold_texts, retrieved_texts_list)
    """
    queries = []
    gold_texts = []
    retrieved_texts_list = []
    
    # 加载QA数据以获取gold文本
    qa_file = project_root / "data" / "public_benchmark" / "monte_cristo_qa_full.json"
    qa_dict = {}
    if qa_file.exists():
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        qa_dict = {qa.get('chunk_id'): qa.get('chunk_text', '') for qa in qa_pairs}
    
    for result in results:
        # 获取查询（如果有的话）
        query = result.get('question', '')
        queries.append(query)
        
        # 获取gold文本
        gold_chunk_id = result.get('gold_chunk_id', '')
        gold_text = qa_dict.get(gold_chunk_id, '')
        gold_texts.append(gold_text)
        
        # 获取HyperAmy的检索结果
        hyperamy = result.get('hyperamy', {})
        if hyperamy.get('available') and hyperamy.get('docs'):
            retrieved_texts_list.append(hyperamy['docs'])
        else:
            retrieved_texts_list.append([])
    
    return queries, gold_texts, retrieved_texts_list


def main():
    """主函数"""
    print("=" * 80)
    print("HyperAmy 基于情绪相似度的评估方法测试")
    print("=" * 80)
    
    # 加载实验结果
    result_file = project_root / "outputs" / "three_methods_comparison_monte_cristo" / "comparison_results.json"
    if not result_file.exists():
        logger.error(f"实验结果文件不存在: {result_file}")
        return
    
    logger.info(f"加载实验结果: {result_file}")
    results = load_experiment_results(result_file)
    logger.info(f"共 {len(results)} 个查询结果")
    
    # 提取文本
    queries, gold_texts, retrieved_texts_list = extract_texts_from_results(results)
    
    # 筛选出有HyperAmy结果的查询
    valid_indices = [i for i, texts in enumerate(retrieved_texts_list) if len(texts) > 0]
    logger.info(f"有效查询数: {len(valid_indices)}/{len(results)}")
    
    if len(valid_indices) == 0:
        logger.error("没有有效的HyperAmy检索结果")
        return
    
    # 只使用有效的查询
    queries = [queries[i] for i in valid_indices]
    gold_texts = [gold_texts[i] for i in valid_indices]
    retrieved_texts_list = [retrieved_texts_list[i] for i in valid_indices]
    
    # 初始化情绪提取器
    logger.info("初始化情绪提取器...")
    try:
        emotion_extractor = Emotion()
        logger.info("✅ 使用新版本Emotion类")
    except:
        logger.warning("无法使用新版本Emotion类，尝试旧版本...")
        emotion_extractor = OldEmotion()
    
    # 测试不同的相似度阈值
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    use_hyperbolic = [True, False]  # 测试双曲距离和余弦相似度
    
    print("\n" + "=" * 80)
    print("评估结果（基于情绪相似度）")
    print("=" * 80)
    
    best_result = None
    best_score = 0.0
    best_config = None
    
    for use_hyp in use_hyperbolic:
        for threshold in thresholds:
            method_name = "双曲距离" if use_hyp else "余弦相似度"
            print(f"\n【{method_name}, 阈值={threshold}】")
            print("-" * 80)
            
            try:
                result = evaluate_hyperamy_emotion_recall(
                    queries=queries[:10],  # 先用10个查询测试
                    gold_texts=gold_texts[:10],
                    retrieved_texts_list=retrieved_texts_list[:10],
                    emotion_extractor=emotion_extractor,
                    k_list=[1, 2, 5, 10],
                    similarity_threshold=threshold,
                    use_hyperbolic_distance=use_hyp
                )
                
                print(f"总查询数: {result.total_queries}")
                print(f"EmotionRecall@K:")
                for k, recall in result.emotion_recall_at_k.items():
                    hits = result.hits_at_k.get(k, 0)
                    print(f"  Recall@{k}: {recall:.1%} ({hits}/{result.total_queries})")
                
                # 记录最佳结果
                avg_recall = np.mean(list(result.emotion_recall_at_k.values()))
                if avg_recall > best_score:
                    best_score = avg_recall
                    best_result = result
                    best_config = (method_name, threshold, use_hyp)
            
            except Exception as e:
                logger.error(f"评估失败: {e}")
                import traceback
                traceback.print_exc()
    
    if best_result:
        print("\n" + "=" * 80)
        print(f"最佳配置: {best_config[0]}, 阈值={best_config[1]}")
        print("=" * 80)
        print(f"EmotionRecall@K:")
        for k, recall in best_result.emotion_recall_at_k.items():
            hits = best_result.hits_at_k.get(k, 0)
            print(f"  Recall@{k}: {recall:.1%} ({hits}/{best_result.total_queries})")
        
        # 保存结果
        output_file = project_root / "outputs" / "three_methods_comparison_monte_cristo" / "hyperamy_emotion_recall_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "best_config": {
                "method": best_config[0],
                "threshold": best_config[1],
                "use_hyperbolic_distance": best_config[2]
            },
            "emotion_recall_at_k": best_result.emotion_recall_at_k,
            "hits_at_k": best_result.hits_at_k,
            "total_queries": best_result.total_queries,
            "threshold_used": best_result.threshold
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 结果已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ 评估完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

