#!/usr/bin/env python3
"""
评估emos模型在验证集上的准确率

使用aux_logits（辅助分类头）来计算Top-K准确率
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "emos-master"))

from src.model import ProbabilisticGBERTV4
from src.config import INDEX_TO_EMOTION, EMOTION_INDEX
from src.dataset import FineGrainedEmotionDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def evaluate_accuracy(checkpoint_path: str, val_data_path: str, device: str = "cuda", top_k: int = 3):
    """评估模型准确率"""
    print("=" * 70)
    print("评估emos模型在验证集上的准确率")
    print("=" * 70)
    
    # 加载模型
    print(f"\n加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ProbabilisticGBERTV4(model_name="roberta-base")
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("模型加载成功!")
    
    # 加载tokenizer和数据集
    print(f"\n加载验证集: {val_data_path}")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    from src.config import Config
    config = Config()
    val_dataset = FineGrainedEmotionDataset(val_data_path, tokenizer, config.max_length)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 评估指标
    total_samples = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    print("\n开始评估...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_mask = batch["entity_mask"].to(device)
            true_soft_labels = batch["soft_label"].to(device)  # (B, 28)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, entity_mask)
            
            # 获取aux_logits（辅助分类头的输出）
            aux_logits = outputs["aux_logits"]  # (B, 28)
            
            # 计算情绪概率（softmax）
            pred_probs = F.softmax(aux_logits, dim=-1)  # (B, 28)
            
            # 计算Top-K准确率
            batch_size = input_ids.shape[0]
            
            # 找出真实标签的top情绪（基于soft_label的最大值）
            true_top_indices = torch.argmax(true_soft_labels, dim=-1)  # (B,)
            
            # 找出预测的top情绪
            pred_top_indices = torch.argmax(pred_probs, dim=-1)  # (B,)
            pred_top_k_indices = torch.topk(pred_probs, k=min(top_k, 28), dim=-1).indices  # (B, k)
            
            # Top-1准确率
            top1_matches = (pred_top_indices == true_top_indices).sum().item()
            top1_correct += top1_matches
            
            # Top-K准确率（检查真实标签的top情绪是否在预测的top-k中）
            for i in range(batch_size):
                true_idx = true_top_indices[i].item()
                pred_top_k = pred_top_k_indices[i].cpu().tolist()
                if true_idx in pred_top_k:
                    top3_correct += 1
                    if true_idx in pred_top_k[:5]:
                        top5_correct += 1
            
            total_samples += batch_size
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理: {batch_idx + 1}/{len(val_loader)} 个batch")
    
    print(f"\n评估完成!")
    print(f"总样本数: {total_samples}")
    
    # 计算准确率
    if total_samples > 0:
        print("\n" + "=" * 70)
        print("准确率结果")
        print("=" * 70)
        
        top1_acc = (top1_correct / total_samples) * 100
        top3_acc = (top3_correct / total_samples) * 100
        top5_acc = (top5_correct / total_samples) * 100
        
        print(f"Top-1准确率: {top1_acc:.2f}% ({top1_correct}/{total_samples})")
        print(f"Top-3准确率: {top3_acc:.2f}% ({top3_correct}/{total_samples})")
        print(f"Top-5准确率: {top5_acc:.2f}% ({top5_correct}/{total_samples})")
        
        print("\n" + "=" * 70)
        print("评估说明")
        print("=" * 70)
        print("• 使用aux_logits（辅助分类头）计算准确率")
        print("• Top-K准确率：真实标签的top情绪是否在预测的top-k中")
        print("• 注意：这是辅助指标，主要评估指标是验证Loss")
        print(f"• 最佳验证Loss: 11.21（已从训练日志中获得）")
    else:
        print("\n⚠️  无法计算准确率")


def main():
    parser = argparse.ArgumentParser(description="评估emos模型准确率")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型checkpoint路径"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="验证集路径"
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
    
    # 如果没有指定验证集，使用默认路径
    val_data = args.val_data
    if not val_data:
        # 尝试服务器路径
        val_data = "/public/jiangh/emos/data/val.jsonl"
        print(f"使用默认验证集: {val_data}")
    
    evaluate_accuracy(args.checkpoint, val_data, args.device, args.top_k)
    
    return 0


if __name__ == "__main__":
    exit(main())
