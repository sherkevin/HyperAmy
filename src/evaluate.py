# -*- coding: utf-8 -*-
"""
GoT 实验自动评测系统

使用 LLM-as-a-Judge 评估 Baseline vs HyperAmy 的答案质量
"""

import json
import os
from typing import List, Dict
from tqdm import tqdm
import logging
import re

from llm import create_client
from llm.config import DEFAULT_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_experiment_results(file_path: str) -> List[Dict]:
    """加载实验结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def judge_answer(question: str, 
                gold_answer: str,
                baseline_answer: str,
                hyperamy_answer: str,
                baseline_context: List[str],
                hyperamy_context: List[str],
                client,
                model_name: str = DEFAULT_MODEL,
                max_retries: int = 3) -> Dict:
    """
    使用 LLM 裁判评估答案
    
    Args:
        question: 问题
        gold_answer: 标准答案
        baseline_answer: Baseline 生成的答案
        hyperamy_answer: HyperAmy 生成的答案
        baseline_context: Baseline 检索到的上下文
        hyperamy_context: HyperAmy 检索到的上下文
        client: LLM 客户端
        model_name: 模型名称
        max_retries: 最大重试次数
        
    Returns:
        Dict: 评估结果 {'winner': 'B' or 'C' or 'Tie', 'reason': '...', 'baseline_score': float, 'hyperamy_score': float}
    """
    baseline_context_text = "\n".join([f"[文档 {i+1}]\n{doc[:500]}..." if len(doc) > 500 else f"[文档 {i+1}]\n{doc}" 
                                      for i, doc in enumerate(baseline_context)])
    hyperamy_context_text = "\n".join([f"[文档 {i+1}]\n{doc[:500]}..." if len(doc) > 500 else f"[文档 {i+1}]\n{doc}" 
                                      for i, doc in enumerate(hyperamy_context)])
    
    prompt = f"""你是一位专业的文学分析专家和评估者。请评估两个模型对以下问题的回答质量。

问题：{question}

标准答案（Ground Truth）：{gold_answer}

---

模型 B (Baseline - 标准语义检索) 的回答：
上下文：
{baseline_context_text}

答案：{baseline_answer}

---

模型 C (HyperAmy - 情感增强检索) 的回答：
上下文：
{hyperamy_context_text}

答案：{hyperamy_answer}

---

请按照以下标准评估：

1. **事实准确性**：哪个答案更准确地包含了标准答案中的关键事实？
2. **危机感知**：哪个答案更好地捕捉了事件的"紧迫性"、"情感权重"或"危机信号"？
3. **上下文相关性**：哪个答案的上下文更相关，更能支持回答？

请以 JSON 格式输出评估结果：
{{
    "winner": "B" 或 "C" 或 "Tie",
    "reason": "详细的评估理由",
    "baseline_score": 0.0-1.0 之间的分数,
    "hyperamy_score": 0.0-1.0 之间的分数,
    "fact_accuracy": {{"B": 0.0-1.0, "C": 0.0-1.0}},
    "crisis_perception": {{"B": 0.0-1.0, "C": 0.0-1.0}},
    "context_relevance": {{"B": 0.0-1.0, "C": 0.0-1.0}}
}}"""

    for attempt in range(max_retries):
        try:
            response = client.get_answer(
                prompt,
                max_tokens=500,
                temperature=0.3,  # 低温度保证一致性
                mode="normal"
            )
            
            # 尝试解析 JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # 验证结果格式
                if 'winner' in result and result['winner'] in ['B', 'C', 'Tie']:
                    # 确保所有字段都存在
                    if 'baseline_score' not in result:
                        result['baseline_score'] = result.get('fact_accuracy', {}).get('B', 0.5)
                    if 'hyperamy_score' not in result:
                        result['hyperamy_score'] = result.get('fact_accuracy', {}).get('C', 0.5)
                    
                    return result
            
            # 如果解析失败，尝试简单提取
            winner = None
            if '"winner"' in response or "'winner'" in response:
                if '"C"' in response or "'C'" in response:
                    winner = 'C'
                elif '"B"' in response or "'B'" in response:
                    winner = 'B'
                else:
                    winner = 'Tie'
            
            if winner:
                return {
                    'winner': winner,
                    'reason': response[:200] + '...' if len(response) > 200 else response,
                    'baseline_score': 0.5,
                    'hyperamy_score': 0.5
                }
            
            logger.warning(f"Failed to parse judge response, attempt {attempt + 1}")
            
        except Exception as e:
            logger.warning(f"Error in judge evaluation, attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to evaluate after {max_retries} attempts")
                return {
                    'winner': 'Tie',
                    'reason': f'Evaluation failed: {e}',
                    'baseline_score': 0.5,
                    'hyperamy_score': 0.5
                }
    
    return {
        'winner': 'Tie',
        'reason': 'Failed to get valid evaluation',
        'baseline_score': 0.5,
        'hyperamy_score': 0.5
    }


def evaluate_experiment(
    experiment_file: str = "results/experiment_run_v1.json",
    output_file: str = "results/evaluation_results.json",
    model_name: str = DEFAULT_MODEL
):
    """
    评估实验结果
    
    Args:
        experiment_file: 实验结果文件路径
        output_file: 输出文件路径
        model_name: LLM 模型名称（用于裁判）
    """
    # 1. 加载实验结果
    logger.info(f"Loading experiment results from {experiment_file}...")
    results = load_experiment_results(experiment_file)
    logger.info(f"Loaded {len(results)} results")
    
    # 2. 初始化 LLM 客户端（裁判）
    logger.info(f"Initializing judge LLM client with model: {model_name}...")
    judge_client = create_client(model_name=model_name, mode="normal")
    
    # 3. 评估每个结果
    logger.info("Evaluating results...")
    evaluations = []
    
    for result in tqdm(results, desc="Evaluating"):
        question = result['question']
        gold_answer = result['gold_answer']
        baseline_answer = result['baseline']['answer']
        hyperamy_answer = result['hyperamy']['answer']
        baseline_context = result['baseline']['context']
        hyperamy_context = result['hyperamy']['context']
        
        # 使用 LLM 裁判评估
        judgment = judge_answer(
            question=question,
            gold_answer=gold_answer,
            baseline_answer=baseline_answer,
            hyperamy_answer=hyperamy_answer,
            baseline_context=baseline_context,
            hyperamy_context=hyperamy_context,
            client=judge_client,
            model_name=model_name
        )
        
        evaluation = {
            'question': question,
            'gold_answer': gold_answer,
            'baseline_answer': baseline_answer,
            'hyperamy_answer': hyperamy_answer,
            'baseline_hit': result['baseline']['hit'],
            'hyperamy_hit': result['hyperamy']['hit'],
            'judgment': judgment
        }
        
        evaluations.append(evaluation)
    
    # 4. 计算统计指标
    baseline_wins = sum(1 for e in evaluations if e['judgment']['winner'] == 'B')
    hyperamy_wins = sum(1 for e in evaluations if e['judgment']['winner'] == 'C')
    ties = sum(1 for e in evaluations if e['judgment']['winner'] == 'Tie')
    
    baseline_hit_rate = sum(1 for e in evaluations if e['baseline_hit']) / len(evaluations)
    hyperamy_hit_rate = sum(1 for e in evaluations if e['hyperamy_hit']) / len(evaluations)
    
    avg_baseline_score = sum(e['judgment']['baseline_score'] for e in evaluations) / len(evaluations)
    avg_hyperamy_score = sum(e['judgment']['hyperamy_score'] for e in evaluations) / len(evaluations)
    
    summary = {
        'total_questions': len(evaluations),
        'baseline_wins': baseline_wins,
        'hyperamy_wins': hyperamy_wins,
        'ties': ties,
        'hyperamy_win_rate': hyperamy_wins / len(evaluations),
        'baseline_hit_rate': baseline_hit_rate,
        'hyperamy_hit_rate': hyperamy_hit_rate,
        'avg_baseline_score': avg_baseline_score,
        'avg_hyperamy_score': avg_hyperamy_score,
        'evaluations': evaluations
    }
    
    # 5. 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {output_file}")
    
    # 6. 打印统计信息
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Total questions: {len(evaluations)}")
    logger.info(f"Baseline wins: {baseline_wins} ({100*baseline_wins/len(evaluations):.1f}%)")
    logger.info(f"HyperAmy wins: {hyperamy_wins} ({100*hyperamy_wins/len(evaluations):.1f}%)")
    logger.info(f"Ties: {ties} ({100*ties/len(evaluations):.1f}%)")
    logger.info(f"HyperAmy Win Rate: {100*hyperamy_wins/len(evaluations):.1f}%")
    logger.info(f"Baseline Hit Rate: {100*baseline_hit_rate:.1f}%")
    logger.info(f"HyperAmy Hit Rate: {100*hyperamy_hit_rate:.1f}%")
    logger.info(f"Average Baseline Score: {avg_baseline_score:.3f}")
    logger.info(f"Average HyperAmy Score: {avg_hyperamy_score:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GoT 实验自动评测")
    parser.add_argument("--experiment", type=str, default="results/experiment_run_v1.json", help="实验结果文件路径")
    parser.add_argument("--output", type=str, default="results/evaluation_results.json", help="输出文件路径")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM 模型名称（裁判）")
    
    args = parser.parse_args()
    
    evaluate_experiment(
        experiment_file=args.experiment,
        output_file=args.output,
        model_name=args.model
    )

