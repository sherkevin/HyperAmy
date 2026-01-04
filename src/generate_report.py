# -*- coding: utf-8 -*-
"""
GoT 实验结果分析与可视化报告生成

生成包含案例研究、数据曲线和分析的完整报告
"""

import json
import os
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_evaluation_results(file_path: str) -> Dict:
    """加载评测结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_experiment_results(file_path: str) -> List[Dict]:
    """加载实验结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_analysis_report(
    evaluation_file: str = "results/evaluation_results.json",
    experiment_file: str = "results/experiment_run_v1.json",
    output_file: str = "results/analysis_report.md"
):
    """
    生成分析报告
    
    Args:
        evaluation_file: 评测结果文件路径
        experiment_file: 实验结果文件路径
        output_file: 输出报告文件路径
    """
    # 1. 加载数据
    logger.info("Loading evaluation and experiment results...")
    eval_results = load_evaluation_results(evaluation_file)
    exp_results = load_experiment_results(experiment_file)
    
    # 2. 生成报告
    report_lines = []
    
    # 标题
    report_lines.append("# HyperAmy GoT 实验结果分析报告\n")
    report_lines.append("## 实验概述\n")
    report_lines.append(f"本报告展示了 HyperAmy（情感增强检索）与 Baseline（标准语义检索）在《权力的游戏》数据集上的对比实验结果。\n")
    
    # 核心指标
    report_lines.append("## 核心指标\n")
    report_lines.append(f"- **总问题数**: {eval_results['total_questions']}")
    report_lines.append(f"- **HyperAmy 胜率**: {eval_results['hyperamy_win_rate']*100:.1f}% ({eval_results['hyperamy_wins']}/{eval_results['total_questions']})")
    report_lines.append(f"- **Baseline 胜率**: {eval_results['baseline_wins']/eval_results['total_questions']*100:.1f}% ({eval_results['baseline_wins']}/{eval_results['total_questions']})")
    report_lines.append(f"- **平局**: {eval_results['ties']} ({eval_results['ties']/eval_results['total_questions']*100:.1f}%)")
    report_lines.append(f"- **Baseline 检索命中率**: {eval_results['baseline_hit_rate']*100:.1f}%")
    report_lines.append(f"- **HyperAmy 检索命中率**: {eval_results['hyperamy_hit_rate']*100:.1f}%")
    report_lines.append(f"- **平均 Baseline 分数**: {eval_results['avg_baseline_score']:.3f}")
    report_lines.append(f"- **平均 HyperAmy 分数**: {eval_results['avg_hyperamy_score']:.3f}\n")
    
    # 案例研究
    report_lines.append("## 案例研究\n")
    
    # 找出 HyperAmy 获胜的案例
    hyperamy_wins = [e for e in eval_results['evaluations'] if e['judgment']['winner'] == 'C']
    baseline_wins = [e for e in eval_results['evaluations'] if e['judgment']['winner'] == 'B']
    
    if hyperamy_wins:
        report_lines.append("### 案例 1: HyperAmy 获胜案例\n")
        case = hyperamy_wins[0]
        report_lines.append(f"**问题**: {case['question']}\n")
        report_lines.append(f"**标准答案**: {case['gold_answer']}\n")
        report_lines.append(f"**Baseline 答案**: {case['baseline_answer'][:200]}...\n")
        report_lines.append(f"**HyperAmy 答案**: {case['hyperamy_answer'][:200]}...\n")
        report_lines.append(f"**评估理由**: {case['judgment']['reason']}\n")
        report_lines.append(f"**Baseline 命中**: {'是' if case['baseline_hit'] else '否'}")
        report_lines.append(f"**HyperAmy 命中**: {'是' if case['hyperamy_hit'] else '否'}\n")
    
    if baseline_wins:
        report_lines.append("### 案例 2: Baseline 获胜案例\n")
        case = baseline_wins[0]
        report_lines.append(f"**问题**: {case['question']}\n")
        report_lines.append(f"**标准答案**: {case['gold_answer']}\n")
        report_lines.append(f"**Baseline 答案**: {case['baseline_answer'][:200]}...\n")
        report_lines.append(f"**HyperAmy 答案**: {case['hyperamy_answer'][:200]}...\n")
        report_lines.append(f"**评估理由**: {case['judgment']['reason']}\n")
        report_lines.append(f"**Baseline 命中**: {'是' if case['baseline_hit'] else '否'}")
        report_lines.append(f"**HyperAmy 命中**: {'是' if case['hyperamy_hit'] else '否'}\n")
    
    # 按质量分数分析
    report_lines.append("## 按质量分数分析\n")
    
    # 找出高质量问题（mass 高的）
    high_mass_cases = []
    for eval_item, exp_item in zip(eval_results['evaluations'], exp_results):
        # 尝试从实验数据中获取 mass 信息
        if 'hyperamy' in exp_item and 'retrieved_chunk_ids' in exp_item['hyperamy']:
            # 这里可以进一步分析，暂时简化
            pass
    
    # 统计不同质量区间的表现
    report_lines.append("### 检索命中率对比\n")
    report_lines.append(f"- Baseline 命中率: {eval_results['baseline_hit_rate']*100:.1f}%")
    report_lines.append(f"- HyperAmy 命中率: {eval_results['hyperamy_hit_rate']*100:.1f}%")
    if eval_results['hyperamy_hit_rate'] > eval_results['baseline_hit_rate']:
        if eval_results['baseline_hit_rate'] > 0:
            improvement = (eval_results['hyperamy_hit_rate'] - eval_results['baseline_hit_rate']) / eval_results['baseline_hit_rate'] * 100
            report_lines.append(f"- **HyperAmy 相对提升**: {improvement:.1f}%\n")
        else:
            improvement = eval_results['hyperamy_hit_rate'] * 100
            report_lines.append(f"- **HyperAmy 绝对提升**: +{improvement:.1f}% (Baseline为0%)\n")
    
    # 结论
    report_lines.append("## 结论\n")
    
    if eval_results['hyperamy_win_rate'] > 0.6:
        report_lines.append("✅ **HyperAmy 在情感增强检索任务上显著优于 Baseline**，胜率超过 60%，证明了情感调制检索的有效性。\n")
    elif eval_results['hyperamy_win_rate'] > 0.5:
        report_lines.append("✅ **HyperAmy 在情感增强检索任务上略优于 Baseline**，胜率超过 50%，显示了情感调制检索的潜力。\n")
    else:
        report_lines.append("⚠️ **HyperAmy 在本次实验中未显著优于 Baseline**，可能需要进一步调优参数或改进方法。\n")
    
    if eval_results['hyperamy_hit_rate'] > eval_results['baseline_hit_rate']:
        report_lines.append("✅ **HyperAmy 的检索命中率高于 Baseline**，说明情感增强检索能够更好地定位关键信息。\n")
    
    report_lines.append("\n## 技术细节\n")
    report_lines.append("- **检索方法**: 庞加莱畸变公式 `warped_dist = cosine_dist / (1 + beta * mass)`")
    report_lines.append("- **Beta 参数**: 10")
    report_lines.append("- **质量阈值**: 0.8")
    report_lines.append("- **检索数量**: Top-3")
    
    # 保存报告
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Analysis report generated and saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成实验结果分析报告")
    parser.add_argument("--evaluation", type=str, default="results/evaluation_results.json", help="评测结果文件路径")
    parser.add_argument("--experiment", type=str, default="results/experiment_run_v1.json", help="实验结果文件路径")
    parser.add_argument("--output", type=str, default="results/analysis_report.md", help="输出报告文件路径")
    
    args = parser.parse_args()
    
    generate_analysis_report(
        evaluation_file=args.evaluation,
        experiment_file=args.experiment,
        output_file=args.output
    )

