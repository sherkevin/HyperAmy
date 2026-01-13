#!/usr/bin/env python3
"""
二阶段训练：对比学习训练脚本

使用筛选出的难例数据，进行对比学习训练。
目标：让排序靠前的context距离Q近，排序靠后的context距离Q远。
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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


class HardNegativeDataset(Dataset):
    """难例数据集"""
    
    def __init__(self, data_file: Path, max_length: int = 128):
        """
        Args:
            data_file: JSONL文件，每行包含question和contexts列表
            max_length: 最大序列长度
        """
        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        self.max_length = max_length
        logger.info(f"加载了 {len(self.samples)} 个难例样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class RankingLoss(nn.Module):
    """
    Ranking Loss for contrastive learning
    
    目标：对于contexts列表 [c1, c2, ..., cn]（按ground truth顺序），
    希望 sim(Q, c1) > sim(Q, c2) > ... > sim(Q, cn)
    """
    
    def __init__(self, margin: float = 0.1):
        """
        Args:
            margin: margin参数，用于ranking loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        q_embeddings: torch.Tensor,  # (B, 64) Q的句子级embedding
        c_embeddings: torch.Tensor,  # (B, N, 64) Contexts的embeddings (N个contexts)
    ) -> torch.Tensor:
        """
        计算ranking loss
        
        Args:
            q_embeddings: (B, 64) Q的embedding
            c_embeddings: (B, N, 64) N个contexts的embeddings
        
        Returns:
            loss: scalar loss value
        """
        batch_size, num_contexts, embed_dim = c_embeddings.shape
        
        # 计算Q与每个context的相似度 (B, N)
        # q_embeddings: (B, 64), c_embeddings: (B, N, 64)
        # 使用cosine similarity
        q_norm = F.normalize(q_embeddings, p=2, dim=1)  # (B, 64)
        c_norm = F.normalize(c_embeddings, p=2, dim=2)  # (B, N, 64)
        
        similarities = torch.bmm(
            q_norm.unsqueeze(1),  # (B, 1, 64)
            c_norm.transpose(1, 2)  # (B, 64, N)
        ).squeeze(1)  # (B, N)
        
        # Ranking loss: 对于位置i < j，希望 sim(Q, ci) > sim(Q, cj) + margin
        loss = 0.0
        num_pairs = 0
        
        for i in range(num_contexts - 1):
            for j in range(i + 1, num_contexts):
                # sim_i应该大于sim_j + margin
                sim_i = similarities[:, i]  # (B,)
                sim_j = similarities[:, j]  # (B,)
                
                # Margin ranking loss: max(0, margin - (sim_i - sim_j))
                pair_loss = F.relu(self.margin - (sim_i - sim_j))
                loss += pair_loss.mean()
                num_pairs += 1
        
        if num_pairs > 0:
            loss = loss / num_pairs
        
        return loss


def get_sentence_embedding(
    model, tokenizer, text: str, device: str = "cpu", max_length: int = 128
) -> torch.Tensor:
    """
    获取文本的句子级embedding (64维向量)
    
    Args:
        model: ProbabilisticGBERTV4模型
        tokenizer: tokenizer
        text: 输入文本
        device: 设备
        max_length: 最大长度
    
    Returns:
        embedding: (64,) 句子级embedding (mu向量)
    """
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Forward pass to get sentence embedding
    # 注意：训练模式下不使用no_grad()，以支持梯度计算
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    mu = outputs["mu"]  # (1, 64)
    mu = mu.squeeze(0)  # (64,)
    
    return mu


def train_epoch(
    model,
    tokenizer,
    dataloader,
    loss_fn,
    optimizer,
    device: str = "cpu",
    max_length: int = 128
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        questions = batch['question']
        gt_contexts_list = batch['ground_truth_contexts']
        negative_contexts_list = batch['negative_contexts']
        
        batch_size = len(questions)
        
        # 收集所有Q和Contexts的embeddings
        q_embeddings_list = []
        c_embeddings_list = []
        
        for i in range(batch_size):
            question = questions[i]
            gt_contexts = gt_contexts_list[i]
            negative_contexts = negative_contexts_list[i]
            
            # 合并contexts（ground truth在前）
            all_contexts = gt_contexts + negative_contexts
            
            # 获取Q的embedding（训练时需要梯度）
            q_emb = get_sentence_embedding(model, tokenizer, question, device, max_length)
            q_embeddings_list.append(q_emb)
            
            # 获取所有contexts的embeddings（训练时需要梯度）
            c_embs = []
            for context in all_contexts:
                c_emb = get_sentence_embedding(model, tokenizer, context, device, max_length)
                c_embs.append(c_emb)
            
            # Stack contexts (N, 64)
            if c_embs:
                c_embeddings_list.append(torch.stack(c_embs))
            else:
                # 如果没有contexts，跳过这个样本
                continue
        
        if not q_embeddings_list or not c_embeddings_list:
            continue
        
        # Stack to batch tensors
        q_embeddings = torch.stack(q_embeddings_list)  # (B, 64)
        
        # Padding contexts to same length
        max_contexts = max(len(c) for c in c_embeddings_list)
        embed_dim = c_embeddings_list[0].shape[-1]
        
        padded_c_embeddings = []
        for c_embs in c_embeddings_list:
            num_contexts = len(c_embs)
            if num_contexts < max_contexts:
                # Padding with zeros
                padding = torch.zeros(max_contexts - num_contexts, embed_dim, device=device)
                c_embs_padded = torch.cat([c_embs, padding], dim=0)
            else:
                c_embs_padded = c_embs
            padded_c_embeddings.append(c_embs_padded)
        
        c_embeddings = torch.stack(padded_c_embeddings)  # (B, N, 64)
        
        # Compute loss
        loss = loss_fn(q_embeddings, c_embeddings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="二阶段训练：对比学习")
    parser.add_argument(
        "--hard_negatives_file",
        type=str,
        default="outputs/stage2_training/hard_negatives.jsonl",
        help="难例数据文件路径"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="~/Desktop/best_model.pt",
        help="初始模型checkpoint路径（阶段一训练好的模型）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/stage2_training",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备 (cpu/cuda)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="训练epochs数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="学习率"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Ranking loss margin"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="最大序列长度"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("二阶段训练：对比学习")
    logger.info("=" * 70)
    
    # 加载模型
    logger.info("\n【步骤1】加载模型...")
    model_checkpoint_path = Path(args.model_checkpoint).expanduser()
    if not model_checkpoint_path.exists():
        logger.error(f"模型文件不存在: {model_checkpoint_path}")
        return
    
    try:
        # 尝试多种路径导入
        ProbabilisticGBERTV4 = None
        GbertPredictor = None
        
        # 从项目根目录查找
        project_root = Path(__file__).parent.parent
        for emos_dir_name in ["emos", "emos-master"]:
            emos_path = project_root / emos_dir_name
            if emos_path.exists() and (emos_path / "src" / "model.py").exists():
                sys.path.insert(0, str(emos_path))
                try:
                    from src.model import ProbabilisticGBERTV4, GbertPredictor
                    logger.info(f"✅ 从 {emos_dir_name} 导入模型成功")
                    break
                except ImportError:
                    continue
        
        # 如果还是失败，尝试从环境变量指定的路径
        if GbertPredictor is None:
            import os
            emos_env_path = os.environ.get('EMOS_PATH', '')
            if emos_env_path and os.path.exists(emos_env_path):
                sys.path.insert(0, emos_env_path)
                try:
                    from src.model import ProbabilisticGBERTV4, GbertPredictor
                    logger.info(f"✅ 从环境变量 EMOS_PATH={emos_env_path} 导入模型成功")
                except ImportError:
                    pass
        
        if GbertPredictor is None:
            raise ImportError("无法找到emos项目的src.model模块")
        
        predictor = GbertPredictor.from_checkpoint(
            checkpoint_path=str(model_checkpoint_path),
            model_name="roberta-base",
            device=args.device
        )
        model = predictor.model
        tokenizer = predictor.tokenizer
        
        # 切换到训练模式
        model.train()
        logger.info("✅ 模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载数据
    logger.info("\n【步骤2】加载难例数据...")
    hard_negatives_file = Path(args.hard_negatives_file)
    if not hard_negatives_file.exists():
        logger.error(f"难例文件不存在: {hard_negatives_file}")
        return
    
    dataset = HardNegativeDataset(hard_negatives_file, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: {  # 简单的collate function
            'question': [s['question'] for s in x],
            'ground_truth_contexts': [s['ground_truth_contexts'] for s in x],
            'negative_contexts': [s['negative_contexts'] for s in x],
        }
    )
    
    # 初始化loss和optimizer
    loss_fn = RankingLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    logger.info("\n【步骤3】开始训练...")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    logger.info(f"   Margin: {args.margin}")
    
    # 训练循环
    train_losses = []
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        avg_loss = train_epoch(
            model, tokenizer, dataloader, loss_fn, optimizer,
            device=args.device, max_length=args.max_length
        )
        
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # 保存checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"保存checkpoint: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = checkpoint_dir / "best_model_stage2.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"\n✅ 训练完成！")
    logger.info(f"最终模型保存到: {final_model_path}")
    logger.info(f"训练Loss: {train_losses}")


if __name__ == "__main__":
    main()
