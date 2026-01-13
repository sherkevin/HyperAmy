#!/usr/bin/env python3
"""
评估训练好的emos情绪嵌入模型

功能:
- 在验证集上评估模型性能
- 测试句子级和实体级情感分析
- 生成评估报告
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "emos-master"))

from src.model import ProbabilisticGBERTV4, GbertPredictor
from src.utils import get_device


def evaluate_on_validation_set(checkpoint_path: str, val_data_path: str, device: str = "cuda"):
    """在验证集上评估模型"""
    print("=" * 60)
    print("评估模型在验证集上的性能")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {checkpoint_path}")
    predictor = GbertPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name="roberta-base",
        device=device
    )
    print("模型加载成功!")
    
    # 加载验证集
    print(f"\n加载验证集: {val_data_path}")
    val_samples = []
    with open(val_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                val_samples.append(json.loads(line))
    
    print(f"验证集样本数: {len(val_samples)}")
    
    # 评估
    print("\n开始评估...")
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for i, sample in enumerate(val_samples):
        text = sample['text']
        targets = sample.get('targets', [])
        
        if not targets:
            continue
        
        # 对每个target进行评估
        for target in targets:
            span_text = target.get('span_text', '')
            true_soft_label = target.get('soft_label', [])
            
            if not span_text:
                continue
            
            # 预测
            try:
                result = predictor.predict(text, span_text=span_text)
                
                # 提取预测的情绪分布（如果有）
                pred_emotions = result.get('emotions', {})
                
                # 计算top-1情绪
                if pred_emotions:
                    top_emotion = max(pred_emotions.items(), key=lambda x: x[1])
                    # 这里简化处理，实际应该比较soft_label
                
                results.append({
                    'text': text,
                    'span_text': span_text,
                    'prediction': result,
                    'true_soft_label': true_soft_label
                })
                
                total_predictions += 1
                
            except Exception as e:
                print(f"警告: 预测失败 - {e}")
                continue
        
        if (i + 1) % 10 == 0:
            print(f"  已处理: {i + 1}/{len(val_samples)} 个样本")
    
    print(f"\n评估完成!")
    print(f"总预测数: {total_predictions}")
    
    return results


def test_sample_predictions(checkpoint_path: str, device: str = "cuda"):
    """测试一些示例预测"""
    print("\n" + "=" * 60)
    print("示例预测测试")
    print("=" * 60)
    
    # 加载模型
    predictor = GbertPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name="roberta-base",
        device=device
    )
    
    # 测试用例
    test_cases = [
        {
            'text': "I am absolutely furious right now!",
            'entity': None,
            'description': '句子级 - 愤怒情绪'
        },
        {
            'text': "The movie was fantastic but the acting was terrible",
            'entity': "fantastic",
            'description': '实体级 - 正面情绪'
        },
        {
            'text': "I feel so happy and joyful today",
            'entity': None,
            'description': '句子级 - 快乐情绪'
        },
        {
            'text': "This is a terrible disaster",
            'entity': "terrible",
            'description': '实体级 - 负面情绪'
        }
    ]
    
    print("\n")
    for i, test in enumerate(test_cases, 1):
        print(f"测试 {i}: {test['description']}")
        print(f"  文本: {test['text']}")
        if test['entity']:
            print(f"  实体: {test['entity']}")
        
        try:
            result = predictor.predict(test['text'], span_text=test.get('entity'))
            
            print(f"  预测结果:")
            print(f"    主要情绪: {result.get('category', 'N/A')}")
            print(f"    情绪强度: {result.get('intensity', 0):.3f}")
            print(f"    Kappa: {result.get('kappa', 0):.2f}")
            
            emotions = result.get('emotions', {})
            if emotions:
                top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"    Top 3情绪: {', '.join([f'{e}({p:.3f})' for e, p in top_emotions])}")
            
        except Exception as e:
            print(f"  ❌ 预测失败: {e}")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="评估emos情绪嵌入模型")
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
        help="验证集路径（可选）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # 测试示例预测
    test_sample_predictions(args.checkpoint, args.device)
    
    # 如果提供了验证集，进行评估
    if args.val_data and Path(args.val_data).exists():
        results = evaluate_on_validation_set(args.checkpoint, args.val_data, args.device)
        
        # 保存结果
        output_file = Path(args.checkpoint).parent / "evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n评估结果已保存: {output_file}")
    else:
        print("\n未提供验证集，跳过验证集评估")


if __name__ == "__main__":
    main()
