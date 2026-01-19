#!/usr/bin/env python3
"""
使用二阶段训练好的模型重新评估性能
重新计算相似度并评估排序准确性
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "emos-master"))

from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sentence_embedding(model, tokenizer, text: str, device: str = "cpu", max_length: int = 128):
    """获取文本的句子级embedding"""
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        mu = outputs["mu"]  # (1, 64)
        mu = mu.squeeze(0)  # (64,)
    
    return mu


def compute_similarity(q_emb: torch.Tensor, c_emb: torch.Tensor) -> float:
    """计算两个embedding的cosine相似度"""
    similarity = F.cosine_similarity(q_emb.unsqueeze(0), c_emb.unsqueeze(0))
    return similarity.item()


def evaluate_ranking_consistency(predict_order: List[int], ground_truth_order: List[int]) -> Dict[str, Any]:
    """评估预测排序和真实排序的一致性"""
    if len(ground_truth_order) == 1:
        gt_idx = ground_truth_order[0]
        if gt_idx in predict_order:
            position = predict_order.index(gt_idx) + 1
            return {
                'gt_position': position,
                'in_top_1': position == 1,
                'in_top_3': position <= 3,
                'in_top_5': position <= 5,
                'kendall_tau': None
            }
        else:
            return {
                'gt_position': len(predict_order) + 1,
                'in_top_1': False,
                'in_top_3': False,
                'in_top_5': False,
                'kendall_tau': None
            }
    return {}


def main():
    parser = argparse.ArgumentParser(description="使用二阶段训练模型重新评估")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="outputs/stage2_training_remote/checkpoints/best_model_stage2.pt",
        help="二阶段训练好的模型checkpoint路径"
    )
    parser.add_argument(
        "--constructed_data",
        type=str,
        default="outputs/stage2_training_remote/constructed_data.jsonl",
        help="构造的数据文件路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备 (cpu/cuda)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/stage2_training_remote/evaluation_after_training.json",
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    model_checkpoint = Path(args.model_checkpoint)
    if not model_checkpoint.exists():
        logger.error(f"模型文件不存在: {model_checkpoint}")
        return
    
    constructed_data_file = Path(args.constructed_data)
    if not constructed_data_file.exists():
        logger.error(f"构造数据文件不存在: {constructed_data_file}")
        return
    
    logger.info("=" * 70)
    logger.info("使用二阶段训练模型重新评估")
    logger.info("=" * 70)
    
    # 加载模型
    logger.info("\n【步骤1】加载模型...")
    import os
    emos_path = os.environ.get('EMOS_PATH', '')
    if emos_path:
        sys.path.insert(0, emos_path)
    else:
        # 尝试从项目目录查找
        for emos_dir in ["emos", "emos-master"]:
            emos_dir_path = project_root / emos_dir
            if emos_dir_path.exists():
                sys.path.insert(0, str(emos_dir_path))
                break
    
    try:
        from src.model import GbertPredictor
        
        predictor = GbertPredictor.from_checkpoint(
            checkpoint_path=str(model_checkpoint),
            model_name="roberta-base",
            device=args.device
        )
        model = predictor.model
        tokenizer = predictor.tokenizer
        model.eval()
        logger.info("✅ 模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载构造的数据
    logger.info("\n【步骤2】加载构造的数据...")
    constructed_data = []
    with open(constructed_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                constructed_data.append(json.loads(line))
    
    logger.info(f"加载了 {len(constructed_data)} 个样本")
    
    # 重新评估
    logger.info("\n【步骤3】重新计算相似度并评估...")
    evaluation_results = []
    
    for data_item in tqdm(constructed_data, desc="重新评估"):
        question = data_item['question']
        gt_contexts = data_item['ground_truth_contexts']
        negative_contexts = data_item['negative_contexts']
        qa_id = data_item.get('qa_id', data_item.get('gt_chunk_id'))
        
        # 合并所有contexts
        all_contexts = gt_contexts + negative_contexts
        context_indices = list(range(len(all_contexts)))
        
        # 获取Q的embedding
        try:
            q_emb = get_sentence_embedding(model, tokenizer, question, args.device)
        except Exception as e:
            logger.warning(f"处理Q失败 (QA {qa_id}): {e}")
            continue
        
        # 计算每个context的相似度
        context_similarities = []
        for ctx_idx, context_text in enumerate(all_contexts):
            try:
                c_emb = get_sentence_embedding(model, tokenizer, context_text, args.device)
                similarity = compute_similarity(q_emb, c_emb)
                context_similarities.append((ctx_idx, similarity))
            except Exception as e:
                logger.warning(f"处理Context {ctx_idx}失败 (QA {qa_id}): {e}")
                context_similarities.append((ctx_idx, -1.0))
        
        # 按相似度排序
        context_similarities.sort(key=lambda x: x[1], reverse=True)
        predict_order = [idx for idx, _ in context_similarities]
        
        # Ground truth order
        ground_truth_order = [0]  # 第一个context是ground truth
        
        # 评估一致性
        consistency_metrics = evaluate_ranking_consistency(predict_order, ground_truth_order)
        
        evaluation_results.append({
            'qa_id': qa_id,
            'question': question[:100] + '...' if len(question) > 100 else question,
            'predict_order': predict_order,
            'ground_truth_order': ground_truth_order,
            'similarities': {str(idx): sim for idx, sim in context_similarities},
            'consistency_metrics': consistency_metrics
        })
    
    # 统计结果
    logger.info("\n" + "=" * 70)
    logger.info("重新评估结果（使用训练后的模型）")
    logger.info("=" * 70)
    
    top1_count = sum(1 for r in evaluation_results if r['consistency_metrics'].get('in_top_1'))
    top3_count = sum(1 for r in evaluation_results if r['consistency_metrics'].get('in_top_3'))
    top5_count = sum(1 for r in evaluation_results if r['consistency_metrics'].get('in_top_5'))
    
    logger.info(f"Top-1准确率: {top1_count}/{len(evaluation_results)} ({100*top1_count/len(evaluation_results):.1f}%)")
    logger.info(f"Top-3准确率: {top3_count}/{len(evaluation_results)} ({100*top3_count/len(evaluation_results):.1f}%)")
    logger.info(f"Top-5准确率: {top5_count}/{len(evaluation_results)} ({100*top5_count/len(evaluation_results):.1f}%)")
    
    # 保存结果
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存到: {output_file}")
    
    logger.info("\n✅ 重新评估完成！")


if __name__ == "__main__":
    main()
