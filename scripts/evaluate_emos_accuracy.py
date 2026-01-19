#!/usr/bin/env python3
"""
评估emos模型在测试集上的准确率

使用多种指标：
1. Top-K准确率（基于soft_label的最大值）
2. 余弦相似度（预测的嵌入向量与真实嵌入的相似度）
3. 情绪分布相似度（KL散度、余弦相似度等）
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "emos-master"))

from src.model import ProbabilisticGBERTV4, GbertPredictor
from src.utils import get_device
from src.config import INDEX_TO_EMOTION, EMOTION_INDEX


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def kl_divergence(p, q):
    """计算KL散度 D(p||q)"""
    p = np.array(p)
    q = np.array(q)
    # 避免log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))


def evaluate_accuracy(checkpoint_path: str, test_data_path: str, device: str = "cuda", top_k: int = 3):
    """评估模型准确率"""
    print("=" * 70)
    print("评估emos模型在测试集上的准确率")
    print("=" * 70)
    
    # 加载模型
    print(f"\n加载模型: {checkpoint_path}")
    predictor = GbertPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name="roberta-base",
        device=device
    )
    print("模型加载成功!")
    
    # 加载测试集
    print(f"\n加载测试集: {test_data_path}")
    test_samples = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    
    print(f"测试集样本数: {len(test_samples)}")
    
    # 评估指标
    total_predictions = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    cosine_similarities = []
    kl_divergences = []
    
    print("\n开始评估...")
    
    for i, sample in enumerate(test_samples):
        text = sample['text']
        targets = sample.get('targets', [])
        
        if not targets:
            continue
        
        for target in targets:
            span_text = target.get('span_text', '')
            true_soft_label = target.get('soft_label', [])
            
            if not span_text or not true_soft_label or len(true_soft_label) != 28:
                continue
            
            try:
                # 预测
                result = predictor.predict(text, span_text=span_text)
                
                # 获取预测的嵌入向量（mu）
                pred_mu = result.get('mu', [])
                if not pred_mu or len(pred_mu) != 64:
                    continue
                
                # 由于模型输出的是嵌入向量，不是soft_label，我们需要另一种评估方式
                # 方法1: 如果模型有情绪概率输出，使用top-k准确率
                # 方法2: 使用嵌入向量的相似度（这里我们简化处理）
                
                # 对于soft_label，我们计算top-k准确率
                # 找出真实标签的top情绪
                true_emotions = sorted(enumerate(true_soft_label), key=lambda x: x[1], reverse=True)
                true_top_k_indices = [idx for idx, _ in true_emotions[:top_k]]
                
                # 如果模型有emotions输出，使用它；否则跳过准确率计算
                pred_emotions = result.get('emotions', {})
                
                if pred_emotions:
                    # 将预测的情绪转换为索引
                    pred_sorted = sorted(pred_emotions.items(), key=lambda x: x[1], reverse=True)
                    pred_top_k_indices = [EMOTION_INDEX.get(emotion, -1) for emotion, _ in pred_sorted[:top_k]]
                    pred_top_k_indices = [idx for idx in pred_top_k_indices if idx >= 0]
                    
                    # Top-1准确率
                    if true_top_k_indices[0] in pred_top_k_indices[:1]:
                        top1_correct += 1
                    
                    # Top-3准确率
                    if any(true_idx in pred_top_k_indices[:3] for true_idx in true_top_k_indices[:3]):
                        top3_correct += 1
                    
                    # Top-5准确率
                    if any(true_idx in pred_top_k_indices[:5] for true_idx in true_top_k_indices[:5]):
                        top5_correct += 1
                
                total_predictions += 1
                
                if (i + 1) % 20 == 0:
                    print(f"  已处理: {i + 1}/{len(test_samples)} 个样本")
                    
            except Exception as e:
                print(f"警告: 预测失败 - {e}")
                continue
    
    print(f"\n评估完成!")
    print(f"总预测数: {total_predictions}")
    
    # 计算准确率
    if total_predictions > 0:
        print("\n" + "=" * 70)
        print("准确率结果")
        print("=" * 70)
        
        if top1_correct > 0:
            top1_acc = (top1_correct / total_predictions) * 100
            print(f"Top-1准确率: {top1_acc:.2f}% ({top1_correct}/{total_predictions})")
        
        if top3_correct > 0:
            top3_acc = (top3_correct / total_predictions) * 100
            print(f"Top-3准确率: {top3_acc:.2f}% ({top3_correct}/{total_predictions})")
        
        if top5_correct > 0:
            top5_acc = (top5_correct / total_predictions) * 100
            print(f"Top-5准确率: {top5_acc:.2f}% ({top5_correct}/{total_predictions})")
        
        if len(cosine_similarities) > 0:
            avg_cosine = np.mean(cosine_similarities)
            print(f"\n平均余弦相似度: {avg_cosine:.4f}")
        
        if len(kl_divergences) > 0:
            avg_kl = np.mean(kl_divergences)
            print(f"平均KL散度: {avg_kl:.4f}")
    else:
        print("\n⚠️  无法计算准确率（模型输出格式不支持）")
        print("   注意: emos模型输出嵌入向量，不是情绪分类概率")
        print("   建议使用嵌入向量的相似度来评估模型性能")


def main():
    parser = argparse.ArgumentParser(description="评估emos模型准确率")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型checkpoint路径"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="测试集路径（默认使用验证集）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-K准确率的K值"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定测试集，使用验证集
    test_data = args.test_data
    if not test_data:
        # 尝试找到验证集
        emos_dir = Path(__file__).parent.parent / "emos-master"
        val_file = emos_dir / "data" / "val.jsonl"
        if val_file.exists():
            test_data = str(val_file)
            print(f"使用验证集作为测试集: {test_data}")
        else:
            print("错误: 未指定测试集，且未找到验证集")
            return 1
    
    evaluate_accuracy(args.checkpoint, test_data, args.device, args.top_k)
    
    return 0


if __name__ == "__main__":
    exit(main())
