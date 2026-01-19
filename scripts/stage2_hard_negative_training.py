#!/usr/bin/env python3
"""
二阶段训练：难例对齐 (Hard Negative Alignment)

从QA对构造数据，使用模型的hidden_state进行相似度检索排序，
筛选不一致的难例，并进行对比学习训练。
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
from tqdm import tqdm
from collections import defaultdict
import logging
import warnings

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


def load_qa_data(qa_file: Path) -> List[Dict[str, Any]]:
    """加载QA数据"""
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    logger.info(f"加载了 {len(qa_data)} 个QA对")
    return qa_data


def load_chunks_data(chunks_file: Path) -> Dict[int, str]:
    """加载chunks数据，返回chunk_id到文本的映射"""
    chunk_id_to_text = {}
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunk_id = chunk.get('chunk_id') or chunk.get('id')
                text = chunk.get('text') or chunk.get('chunk_text') or chunk.get('input') or chunk.get('content', '')
                if chunk_id is not None and text:
                    chunk_id_to_text[chunk_id] = text
    logger.info(f"加载了 {len(chunk_id_to_text)} 个chunks")
    return chunk_id_to_text


def load_entity_annotations(entity_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    加载实体粒度数据集，返回text到实体标注的映射
    
    Returns:
        Dict[text_normalized, entity_annotation]
    """
    text_to_entities = {}
    
    if not entity_file.exists():
        logger.warning(f"实体标注文件不存在: {entity_file}")
        return text_to_entities
    
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                text = sample.get('text', '').strip()
                targets = sample.get('targets', [])
                
                if text and targets:
                    # 标准化文本（去除多余空格，用于匹配）
                    text_normalized = ' '.join(text.split())
                    text_to_entities[text_normalized] = {
                        'text': text,
                        'targets': targets
                    }
    
    logger.info(f"加载了 {len(text_to_entities)} 个实体标注样本")
    return text_to_entities


def normalize_text(text: str) -> str:
    """标准化文本用于匹配（去除多余空格）"""
    return ' '.join(text.strip().split())


def get_entity_annotation(text: str, text_to_entities: Dict[str, Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """获取文本的实体标注"""
    text_norm = normalize_text(text)
    
    # 精确匹配
    if text_norm in text_to_entities:
        return text_to_entities[text_norm]['targets']
    
    # 模糊匹配（如果文本包含在实体数据集的文本中，或相反）
    for key, value in text_to_entities.items():
        if text_norm in key or key in text_norm:
            if abs(len(text_norm) - len(key)) / max(len(text_norm), len(key), 1) < 0.1:  # 长度差异小于10%
                return value['targets']
    
    return None


def construct_qa_contexts_data(
    qa_data: List[Dict[str, Any]],
    chunk_id_to_text: Dict[int, str],
    num_negative_contexts: int = 5
) -> List[Dict[str, Any]]:
    """
    从QA对构造数据：Q + 多个contexts
    
    Args:
        qa_data: QA数据列表
        chunk_id_to_text: chunk_id到文本的映射
        num_negative_contexts: 负样本context数量
    
    Returns:
        构造的数据列表，每个元素包含：
        - question: 问题文本
        - ground_truth_contexts: [context1, context2, ...] (ground truth顺序)
        - negative_contexts: [context1, context2, ...] (随机负样本)
        - qa_id: QA对的ID
    """
    constructed_data = []
    
    all_chunk_ids = list(chunk_id_to_text.keys())
    
    for qa in tqdm(qa_data, desc="构造数据"):
        question = qa.get('question', '')
        gt_chunk_id = qa.get('chunk_id')
        
        if not question or gt_chunk_id is None:
            continue
        
        gt_context = chunk_id_to_text.get(gt_chunk_id)
        if not gt_context:
            continue
        
        # Ground truth contexts (这里只有一个，后续可以扩展)
        ground_truth_contexts = [gt_context]
        
        # 随机选择负样本contexts
        negative_chunk_ids = [cid for cid in all_chunk_ids if cid != gt_chunk_id]
        if len(negative_chunk_ids) < num_negative_contexts:
            num_negative = len(negative_chunk_ids)
        else:
            num_negative = num_negative_contexts
        
        if num_negative > 0:
            selected_negative_ids = np.random.choice(
                negative_chunk_ids,
                size=num_negative,
                replace=False
            ).tolist()
            negative_contexts = [chunk_id_to_text[cid] for cid in selected_negative_ids]
        else:
            negative_contexts = []
        
        constructed_data.append({
            'question': question,
            'ground_truth_contexts': ground_truth_contexts,
            'negative_contexts': negative_contexts,
            'qa_id': qa.get('chunk_id'),  # 使用chunk_id作为QA ID
            'gt_chunk_id': gt_chunk_id
        })
    
    logger.info(f"构造了 {len(constructed_data)} 个数据样本")
    return constructed_data


def get_token_level_hidden_states(
    model, tokenizer, text: str, device: str = "cpu", max_length: int = 128
) -> torch.Tensor:
    """
    获取文本的token-level hidden states (64维向量)
    
    Args:
        model: ProbabilisticGBERTV4模型
        tokenizer: tokenizer
        text: 输入文本
        device: 设备
        max_length: 最大长度
    
    Returns:
        token_vectors: (L, 64) token级别的hidden states
        encoding: tokenizer encoding结果（包含offset_mapping等）
    """
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Forward pass to get token vectors
    with torch.no_grad():
        # 获取backbone的输出
        outputs = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (1, L, 768)
        
        # Project to 64d (Branch A - semantic_head)
        token_vectors = model.semantic_head(last_hidden)  # (1, L, 64)
        token_vectors = token_vectors.squeeze(0)  # (L, 64)
    
    return token_vectors, encoding


def extract_entity_tokens(
    entity: Dict[str, Any],
    encoding: Dict,
    text: str
) -> Optional[torch.Tensor]:
    """
    从token-level hidden states中提取实体对应的token向量
    
    Args:
        entity: 实体字典，包含span_text, char_start, char_end
        encoding: tokenizer encoding结果
        token_vectors: (L, 64) token vectors
        text: 原始文本
    
    Returns:
        entity_token_mask: (L,) bool tensor，True表示属于该实体的token
    """
    char_start = entity.get('char_start')
    char_end = entity.get('char_end')
    
    if char_start is None or char_end is None:
        return None
    
    # 获取token offsets
    offsets = encoding["offset_mapping"]
    if isinstance(offsets, list):
        if len(offsets) > 0 and isinstance(offsets[0], list):
            offsets = offsets[0]
        offsets = torch.tensor(offsets)
    else:
        if offsets.dim() > 2:
            offsets = offsets.squeeze(0)
    
    # 获取attention mask
    attention_mask = encoding["attention_mask"]
    if isinstance(attention_mask, list):
        if len(attention_mask) > 0 and isinstance(attention_mask[0], list):
            attention_mask = attention_mask[0]
        attention_mask = torch.tensor(attention_mask)
    else:
        if attention_mask.dim() > 1:
            attention_mask = attention_mask.squeeze(0)
    
    # 创建实体mask：token的offset与实体char range重叠
    token_starts = offsets[:, 0]
    token_ends = offsets[:, 1]
    
    entity_mask = (token_starts < char_end) & (token_ends > char_start) & attention_mask.bool()
    
    return entity_mask


def compute_entity_vector(
    token_vectors: torch.Tensor,
    entity_mask: torch.Tensor
) -> torch.Tensor:
    """
    从token vectors中提取实体向量（mean pooling）
    
    Args:
        token_vectors: (L, 64) token vectors
        entity_mask: (L,) bool mask
    
    Returns:
        entity_vector: (64,) entity vector
    """
    if entity_mask.sum() == 0:
        # 如果没有匹配的token，返回零向量
        return torch.zeros(token_vectors.shape[1], device=token_vectors.device)
    
    # Mean pooling over entity tokens
    masked_vectors = token_vectors * entity_mask.unsqueeze(-1).float()
    entity_vector = masked_vectors.sum(dim=0) / entity_mask.sum().float()
    
    return entity_vector


def compute_context_similarity(
    q_token_vectors: torch.Tensor,
    q_encoding: Dict,
    q_entities: List[Dict[str, Any]],
    q_text: str,
    c_token_vectors: torch.Tensor,
    c_encoding: Dict,
    c_entities: List[Dict[str, Any]],
    c_text: str,
    device: str = "cpu"
) -> float:
    """
    计算Q和Context之间的实体级别平均相似度
    
    Args:
        q_token_vectors: (L_q, 64) Q的token vectors
        q_encoding: Q的tokenizer encoding
        q_entities: Q的实体列表
        q_text: Q的原始文本
        c_token_vectors: (L_c, 64) Context的token vectors
        c_encoding: Context的tokenizer encoding
        c_entities: Context的实体列表
        c_text: Context的原始文本
        device: 设备
    
    Returns:
        平均相似度分数
    """
    if not q_entities or not c_entities:
        # 如果没有实体，使用句子级别的相似度（mean pooling）
        q_vec = q_token_vectors.mean(dim=0)
        c_vec = c_token_vectors.mean(dim=0)
        similarity = F.cosine_similarity(q_vec.unsqueeze(0), c_vec.unsqueeze(0))
        return similarity.item()
    
    # 提取Q的所有实体向量
    q_entity_vectors = []
    for entity in q_entities:
        entity_mask = extract_entity_tokens(entity, q_encoding, q_text)
        if entity_mask is not None and entity_mask.sum() > 0:
            entity_vec = compute_entity_vector(q_token_vectors, entity_mask)
            q_entity_vectors.append(entity_vec)
    
    # 提取Context的所有实体向量
    c_entity_vectors = []
    for entity in c_entities:
        entity_mask = extract_entity_tokens(entity, c_encoding, c_text)
        if entity_mask is not None and entity_mask.sum() > 0:
            entity_vec = compute_entity_vector(c_token_vectors, entity_mask)
            c_entity_vectors.append(entity_vec)
    
    if not q_entity_vectors or not c_entity_vectors:
        # Fallback to sentence-level similarity
        q_vec = q_token_vectors.mean(dim=0)
        c_vec = c_token_vectors.mean(dim=0)
        similarity = F.cosine_similarity(q_vec.unsqueeze(0), c_vec.unsqueeze(0))
        return similarity.item()
    
    # 计算所有实体对的相似度
    q_entity_tensor = torch.stack(q_entity_vectors)  # (N_q, 64)
    c_entity_tensor = torch.stack(c_entity_vectors)  # (N_c, 64)
    
    # 计算相似度矩阵 (N_q, N_c)
    similarity_matrix = F.cosine_similarity(
        q_entity_tensor.unsqueeze(1),  # (N_q, 1, 64)
        c_entity_tensor.unsqueeze(0),  # (1, N_c, 64)
        dim=2
    )
    
    # 计算平均相似度
    avg_similarity = similarity_matrix.mean().item()
    
    return avg_similarity


def evaluate_ranking_consistency(
    predict_order: List[int],
    ground_truth_order: List[int]
) -> Dict[str, float]:
    """
    评估预测排序和真实排序的一致性
    
    Args:
        predict_order: 预测的context索引排序（按相似度从高到低）
        ground_truth_order: 真实的context索引排序
    
    Returns:
        一致性指标字典
    """
    # 如果只有一个ground truth context
    if len(ground_truth_order) == 1:
        gt_idx = ground_truth_order[0]
        if gt_idx in predict_order:
            position = predict_order.index(gt_idx) + 1  # 1-indexed
            return {
                'gt_position': position,
                'in_top_1': position == 1,
                'in_top_3': position <= 3,
                'in_top_5': position <= 5,
                'kendall_tau': None  # 不适用
            }
        else:
            return {
                'gt_position': len(predict_order) + 1,
                'in_top_1': False,
                'in_top_3': False,
                'in_top_5': False,
                'kendall_tau': None
            }
    
    # 多个ground truth contexts：使用Kendall's Tau
    from scipy.stats import kendalltau
    
    # 创建位置映射
    predict_ranks = {idx: rank for rank, idx in enumerate(predict_order)}
    gt_ranks = {idx: rank for rank, idx in enumerate(ground_truth_order)}
    
    # 只考虑同时在两个排序中的contexts
    common_indices = set(predict_order) & set(ground_truth_order)
    if len(common_indices) < 2:
        return {
            'gt_position': None,
            'in_top_1': False,
            'in_top_3': False,
            'in_top_5': False,
            'kendall_tau': 0.0
        }
    
    common_indices_list = list(common_indices)
    predict_rank_list = [predict_ranks[idx] for idx in common_indices_list]
    gt_rank_list = [gt_ranks[idx] for idx in common_indices_list]
    
    tau, p_value = kendalltau(predict_rank_list, gt_rank_list)
    
    return {
        'gt_position': None,
        'in_top_1': None,
        'in_top_3': None,
        'in_top_5': None,
        'kendall_tau': tau if not np.isnan(tau) else 0.0,
        'p_value': p_value
    }


def main():
    parser = argparse.ArgumentParser(description="二阶段训练：难例对齐")
    parser.add_argument(
        "--qa_file",
        type=str,
        default="data/benchmarks/instinct_qa.json",
        help="QA数据文件路径"
    )
    parser.add_argument(
        "--chunks_file",
        type=str,
        default="data/processed/got_amygdala.jsonl",
        help="Chunks数据文件路径"
    )
    parser.add_argument(
        "--entity_file",
        type=str,
        default="data/training/entity_granularity/entity_granularity_v2_full.jsonl",
        help="实体标注数据文件路径"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="~/Desktop/best_model.pt",
        help="模型checkpoint路径"
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
        "--num_negative_contexts",
        type=int,
        default=5,
        help="每个QA对的负样本context数量"
    )
    parser.add_argument(
        "--hard_negative_threshold",
        type=float,
        default=3,
        help="难例阈值（ground truth context不在top-K中）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数（用于测试）"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("二阶段训练：难例对齐 - 数据构造和评估")
    logger.info("=" * 70)
    
    # 加载数据
    logger.info("\n【阶段1】加载数据...")
    qa_data = load_qa_data(Path(args.qa_file))
    chunk_id_to_text = load_chunks_data(Path(args.chunks_file))
    text_to_entities = load_entity_annotations(Path(args.entity_file))
    
    # 构造数据
    logger.info("\n【阶段2】构造Q+Contexts数据...")
    constructed_data = construct_qa_contexts_data(
        qa_data, 
        chunk_id_to_text,
        num_negative_contexts=args.num_negative_contexts
    )
    
    if args.max_samples:
        constructed_data = constructed_data[:args.max_samples]
        logger.info(f"限制处理样本数为: {args.max_samples}")
    
    # 保存构造的数据
    constructed_data_file = output_dir / "constructed_data.jsonl"
    with open(constructed_data_file, 'w', encoding='utf-8') as f:
        for item in constructed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"保存构造的数据到: {constructed_data_file}")
    
    logger.info(f"\n✅ 数据准备完成！")
    logger.info(f"   构造了 {len(constructed_data)} 个样本")
    logger.info(f"   每个样本包含 1 个ground truth context 和 {args.num_negative_contexts} 个负样本contexts")
    
    # 展开模型checkpoint路径
    model_checkpoint_path = Path(args.model_checkpoint).expanduser()
    if not model_checkpoint_path.exists():
        logger.error(f"模型文件不存在: {model_checkpoint_path}")
        return
    
    # 阶段3：加载模型并提取hidden states
    logger.info("\n【阶段3】加载模型...")
    try:
        # 尝试多种路径导入
        GbertPredictor = None
        for emos_dir_name in ["emos", "emos-master"]:
            emos_path = project_root / emos_dir_name
            if emos_path.exists() and (emos_path / "src" / "model.py").exists():
                sys.path.insert(0, str(emos_path))
                try:
                    from src.model import GbertPredictor
                    logger.info(f"✅ 从 {emos_dir_name} 导入模型成功")
                    break
                except ImportError as e:
                    continue
        
        # 如果还是失败，尝试从环境变量指定的路径
        if GbertPredictor is None:
            import os
            emos_env_path = os.environ.get('EMOS_PATH', '')
            if emos_env_path and os.path.exists(emos_env_path):
                sys.path.insert(0, emos_env_path)
                try:
                    from src.model import GbertPredictor
                    logger.info(f"✅ 从环境变量 EMOS_PATH={emos_env_path} 导入模型成功")
                except ImportError:
                    pass
        
        if GbertPredictor is None:
            raise ImportError(f"无法找到emos项目的src.model模块。尝试的路径: {[str(p / 'src' / 'model.py') for p in [project_root / 'emos', project_root / 'emos-master'] if (p / 'src' / 'model.py').exists()]}")
        
        predictor = GbertPredictor.from_checkpoint(
            checkpoint_path=str(model_checkpoint_path),
            model_name="roberta-base",
            device=args.device
        )
        model = predictor.model
        tokenizer = predictor.tokenizer
        logger.info("✅ 模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 阶段4：实体标注匹配、相似度计算和排序
    logger.info("\n【阶段4】提取hidden states和计算相似度...")
    evaluation_results = []
    
    for idx, data_item in enumerate(tqdm(constructed_data, desc="处理样本")):
        question = data_item['question']
        gt_contexts = data_item['ground_truth_contexts']
        negative_contexts = data_item['negative_contexts']
        qa_id = data_item['qa_id']
        
        # 合并所有contexts（ground truth + negative）
        all_contexts = gt_contexts + negative_contexts
        context_indices = list(range(len(all_contexts)))  # 0=gt, 1-N=negative
        
        # 提取Q的hidden states和实体
        try:
            q_token_vectors, q_encoding = get_token_level_hidden_states(
                model, tokenizer, question, device=args.device
            )
            q_entities = get_entity_annotation(question, text_to_entities)
            if q_entities is None:
                q_entities = []
        except Exception as e:
            logger.warning(f"处理Q失败 (QA {qa_id}): {e}")
            continue
        
        # 计算每个context的相似度
        context_similarities = []
        context_entities_list = []
        
        for ctx_idx, context_text in enumerate(all_contexts):
            try:
                # 提取context的hidden states和实体
                c_token_vectors, c_encoding = get_token_level_hidden_states(
                    model, tokenizer, context_text, device=args.device
                )
                c_entities = get_entity_annotation(context_text, text_to_entities)
                if c_entities is None:
                    c_entities = []
                
                # 计算相似度
                similarity = compute_context_similarity(
                    q_token_vectors, q_encoding, q_entities, question,
                    c_token_vectors, c_encoding, c_entities, context_text,
                    device=args.device
                )
                
                context_similarities.append((ctx_idx, similarity))
                context_entities_list.append(c_entities)
            except Exception as e:
                logger.warning(f"处理Context {ctx_idx}失败 (QA {qa_id}): {e}")
                context_similarities.append((ctx_idx, -1.0))  # 失败时设为-1
                context_entities_list.append([])
        
        # 按相似度排序
        context_similarities.sort(key=lambda x: x[1], reverse=True)
        predict_order = [idx for idx, _ in context_similarities]
        
        # Ground truth order (只有第一个是ground truth)
        ground_truth_order = [0]  # 第一个context是ground truth
        
        # 评估一致性
        consistency_metrics = evaluate_ranking_consistency(predict_order, ground_truth_order)
        
        evaluation_results.append({
            'qa_id': qa_id,
            'question': question[:100] + '...' if len(question) > 100 else question,
            'predict_order': predict_order,
            'ground_truth_order': ground_truth_order,
            'similarities': {idx: sim for idx, sim in context_similarities},
            'consistency_metrics': consistency_metrics,
            'q_entities_count': len(q_entities),
            'c_entities_count': [len(ents) for ents in context_entities_list]
        })
    
    # 保存评估结果
    evaluation_file = output_dir / "evaluation_results.json"
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    logger.info(f"保存评估结果到: {evaluation_file}")
    
    # 阶段5：筛选难例
    logger.info("\n【阶段5】筛选难例...")
    hard_negatives = []
    
    for eval_result in evaluation_results:
        metrics = eval_result['consistency_metrics']
        # 如果ground truth context不在top-K中，标记为难例
        if metrics.get('gt_position') and metrics['gt_position'] > args.hard_negative_threshold:
            # 找到对应的原始数据
            qa_id = eval_result['qa_id']
            original_data = next((d for d in constructed_data if d['qa_id'] == qa_id), None)
            if original_data:
                hard_negatives.append(original_data)
    
    logger.info(f"筛选出 {len(hard_negatives)} 个难例（共 {len(constructed_data)} 个样本）")
    
    # 保存难例
    hard_negatives_file = output_dir / "hard_negatives.jsonl"
    with open(hard_negatives_file, 'w', encoding='utf-8') as f:
        for item in hard_negatives:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"保存难例到: {hard_negatives_file}")
    
    # 统计信息
    logger.info("\n" + "=" * 70)
    logger.info("评估统计")
    logger.info("=" * 70)
    
    top1_count = sum(1 for r in evaluation_results if r['consistency_metrics'].get('in_top_1'))
    top3_count = sum(1 for r in evaluation_results if r['consistency_metrics'].get('in_top_3'))
    top5_count = sum(1 for r in evaluation_results if r['consistency_metrics'].get('in_top_5'))
    
    logger.info(f"Top-1准确率: {top1_count}/{len(evaluation_results)} ({100*top1_count/len(evaluation_results):.1f}%)")
    logger.info(f"Top-3准确率: {top3_count}/{len(evaluation_results)} ({100*top3_count/len(evaluation_results):.1f}%)")
    logger.info(f"Top-5准确率: {top5_count}/{len(evaluation_results)} ({100*top5_count/len(evaluation_results):.1f}%)")
    logger.info(f"难例数量: {len(hard_negatives)} ({100*len(hard_negatives)/len(evaluation_results):.1f}%)")
    
    logger.info("\n✅ 评估完成！")
    logger.info(f"\n📝 下一步：使用难例数据进行对比学习训练")
    logger.info(f"   难例文件: {hard_negatives_file}")


if __name__ == "__main__":
    main()
