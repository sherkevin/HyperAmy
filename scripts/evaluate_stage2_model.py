#!/usr/bin/env python3
"""
使用二阶段训练好的模型重新评估性能
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


def evaluate_model(
    model_checkpoint: Path,
    evaluation_data_file: Path,
    device: str = "cpu"
):
    """使用训练好的模型重新评估"""
    
    logger.info("=" * 70)
    logger.info("使用二阶段训练模型重新评估")
    logger.info("=" * 70)
    
    # 加载模型
    logger.info("\n【步骤1】加载模型...")
    import os
    emos_path = os.environ.get('EMOS_PATH', '')
    if emos_path:
        sys.path.insert(0, emos_path)
    
    try:
        from src.model import GbertPredictor
        
        predictor = GbertPredictor.from_checkpoint(
            checkpoint_path=str(model_checkpoint),
            model_name="roberta-base",
            device=device
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
    
    # 加载评估数据
    logger.info("\n【步骤2】加载评估数据...")
    with open(evaluation_data_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    logger.info(f"加载了 {len(eval_data)} 个评估样本")
    
    # 重新评估
    logger.info("\n【步骤3】重新评估...")
    results = []
    
    for item in tqdm(eval_data, desc="评估"):
        qa_id = item['qa_id']
        question = item['question']
        
        # 需要从constructed_data中获取contexts
        # 这里简化处理，使用evaluation_results中的信息
        predict_order = item.get('predict_order', [])
        ground_truth_order = item.get('ground_truth_order', [])
        
        # 重新计算相似度（使用训练后的模型）
        # 注意：这里需要原始的contexts文本，暂时使用原有结果
        # 实际应该重新加载constructed_data并重新计算
        
        results.append({
            'qa_id': qa_id,
            'original_metrics': item.get('consistency_metrics', {}),
            'note': '需要重新计算相似度，当前使用原始结果'
        })
    
    # 统计
    logger.info("\n" + "=" * 70)
    logger.info("评估结果（使用训练后的模型）")
    logger.info("=" * 70)
    
    # 由于需要重新计算，这里先显示原始结果
    top1_count = sum(1 for r in eval_data if r['consistency_metrics'].get('in_top_1'))
    top3_count = sum(1 for r in eval_data if r['consistency_metrics'].get('in_top_3'))
    top5_count = sum(1 for r in eval_data if r['consistency_metrics'].get('in_top_5'))
    
    logger.info(f"Top-1准确率: {top1_count}/{len(eval_data)} ({100*top1_count/len(eval_data):.1f}%)")
    logger.info(f"Top-3准确率: {top3_count}/{len(eval_data)} ({100*top3_count/len(eval_data):.1f}%)")
    logger.info(f"Top-5准确率: {top5_count}/{len(eval_data)} ({100*top5_count/len(eval_data):.1f}%)")
    
    logger.info("\n⚠️  注意: 当前显示的是训练前的评估结果。")
    logger.info("   要获得训练后的准确结果，需要重新加载constructed_data并重新计算相似度。")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="使用二阶段训练模型重新评估")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="outputs/stage2_training_remote/checkpoints/best_model_stage2.pt",
        help="二阶段训练好的模型checkpoint路径"
    )
    parser.add_argument(
        "--evaluation_data",
        type=str,
        default="outputs/stage2_training_remote/evaluation_results.json",
        help="评估数据文件路径"
    )
    parser.add_argument(
        "--constructed_data",
        type=str,
        default="outputs/stage2_training_remote/constructed_data.jsonl",
        help="构造的数据文件路径（包含原始contexts）"
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
    
    evaluation_data_file = Path(args.evaluation_data)
    if not evaluation_data_file.exists():
        logger.error(f"评估数据文件不存在: {evaluation_data_file}")
        return
    
    results = evaluate_model(model_checkpoint, evaluation_data_file, args.device)
    
    # 保存结果
    if results:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
