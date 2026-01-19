#!/usr/bin/env python3
"""
分析训练日志，提取关键指标并绘制训练曲线。

支持的功能:
- 解析训练日志文件
- 提取loss、学习率等关键指标
- 绘制训练曲线图
- 生成训练报告

使用方法:
    python scripts/analyze_training_log.py \
        --log_file emos-master/logs/train_full_*.log \
        --output_dir emos-master/logs/analysis
"""

import argparse
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def parse_log_file(log_file: Path) -> Dict[str, Any]:
    """
    解析训练日志文件，提取关键信息。
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        包含训练指标的字典
    """
    metrics = {
        'epoch': [],
        'step': [],
        'total_loss': [],
        'vmf_loss': [],
        'cal_loss': [],
        'aux_loss': [],
        'lr': [],
        'val_loss': [],
        'val_epoch': [],
        'kappa': [],
        'timestamp': []
    }
    
    current_epoch = 0
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析训练步骤信息
            # 示例: "Epoch 1/10, Step 100/500 - Loss: 2.345 - vmf: 1.234 - cal: 0.567 - aux: 0.123 - LR: 2.00e-05"
            train_match = re.search(
                r'Epoch (\d+)/(\d+).*?Step (\d+)/(\d+).*?Loss: ([\d.]+).*?vmf: ([\d.]+).*?cal: ([\d.]+).*?aux: ([\d.]+).*?LR: ([\d.e-]+)',
                line
            )
            
            if train_match:
                epoch = int(train_match.group(1))
                step = int(train_match.group(3))
                total_loss = float(train_match.group(5))
                vmf_loss = float(train_match.group(6))
                cal_loss = float(train_match.group(7))
                aux_loss = float(train_match.group(8))
                lr = float(train_match.group(9))
                
                metrics['epoch'].append(epoch)
                metrics['step'].append(step)
                metrics['total_loss'].append(total_loss)
                metrics['vmf_loss'].append(vmf_loss)
                metrics['cal_loss'].append(cal_loss)
                metrics['aux_loss'].append(aux_loss)
                metrics['lr'].append(lr)
                current_epoch = epoch
            
            # 解析验证信息
            # 示例: "Validation Loss: 1.234, Avg Kappa: 5.678"
            val_match = re.search(
                r'Validation.*?Loss: ([\d.]+).*?Avg Kappa: ([\d.]+)',
                line
            )
            
            if val_match:
                val_loss = float(val_match.group(1))
                kappa = float(val_match.group(2))
                
                metrics['val_loss'].append(val_loss)
                metrics['val_epoch'].append(current_epoch)
                metrics['kappa'].append(kappa)
            
            # 解析最佳模型信息
            # 示例: "Best model saved at step 500"
            best_match = re.search(r'Best model saved', line)
            if best_match:
                metrics['best_step'] = metrics['step'][-1] if metrics['step'] else None
    
    return metrics


def plot_training_curves(metrics: Dict[str, Any], output_dir: Path):
    """
    绘制训练曲线图。
    
    Args:
        metrics: 训练指标字典
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss曲线
    if metrics['step']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves', fontsize=16)
        
        # Total Loss
        axes[0, 0].plot(metrics['step'], metrics['total_loss'], label='Total Loss', alpha=0.7)
        if metrics['val_loss'] and metrics['val_epoch']:
            # 将验证epoch映射到step（简化处理）
            val_steps = [max(0, s - 1) for s in metrics['step'] if s % 500 == 0][:len(metrics['val_loss'])]
            if val_steps:
                axes[0, 0].plot(val_steps, metrics['val_loss'], 'ro-', label='Validation Loss', markersize=5)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Component Losses
        axes[0, 1].plot(metrics['step'], metrics['vmf_loss'], label='VMF Loss', alpha=0.7)
        axes[0, 1].plot(metrics['step'], metrics['cal_loss'], label='Calibration Loss', alpha=0.7)
        axes[0, 1].plot(metrics['step'], metrics['aux_loss'], label='Auxiliary Loss', alpha=0.7)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        if metrics['lr']:
            axes[1, 0].plot(metrics['step'], metrics['lr'], label='Learning Rate', color='green', alpha=0.7)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Validation Loss and Kappa
        if metrics['val_epoch']:
            ax_val = axes[1, 1]
            ax_val.plot(metrics['val_epoch'], metrics['val_loss'], 'ro-', label='Validation Loss', markersize=8)
            ax_val.set_xlabel('Epoch')
            ax_val.set_ylabel('Validation Loss', color='red')
            ax_val.tick_params(axis='y', labelcolor='red')
            ax_val.legend(loc='upper left')
            ax_val.grid(True, alpha=0.3)
            
            if metrics['kappa']:
                ax_kappa = ax_val.twinx()
                ax_kappa.plot(metrics['val_epoch'], metrics['kappa'], 'bs-', label='Avg Kappa', markersize=8)
                ax_kappa.set_ylabel('Avg Kappa', color='blue')
                ax_kappa.tick_params(axis='y', labelcolor='blue')
                ax_kappa.legend(loc='upper right')
        
        plt.tight_layout()
        plot_file = output_dir / 'training_curves.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 训练曲线图已保存: {plot_file}")


def generate_report(metrics: Dict[str, Any], log_file: Path, output_dir: Path):
    """
    生成训练报告。
    
    Args:
        metrics: 训练指标字典
        log_file: 日志文件路径
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# 训练日志分析报告",
        "",
        f"**日志文件**: `{log_file}`",
        f"**生成时间**: {Path(log_file).stat().st_mtime}",
        "",
        "## 训练统计",
        ""
    ]
    
    if metrics['step']:
        total_steps = len(metrics['step'])
        total_epochs = max(metrics['epoch']) if metrics['epoch'] else 0
        
        report_lines.extend([
            f"- **总训练步数**: {total_steps}",
            f"- **总训练轮数**: {total_epochs}",
            f"- **初始Loss**: {metrics['total_loss'][0]:.4f}",
            f"- **最终Loss**: {metrics['total_loss'][-1]:.4f}",
            f"- **Loss下降**: {((metrics['total_loss'][0] - metrics['total_loss'][-1]) / metrics['total_loss'][0] * 100):.2f}%",
        ])
        
        if metrics['val_loss']:
            report_lines.extend([
                "",
                "## 验证统计",
                "",
                f"- **验证次数**: {len(metrics['val_loss'])}",
                f"- **最佳验证Loss**: {min(metrics['val_loss']):.4f}",
                f"- **最终验证Loss**: {metrics['val_loss'][-1]:.4f}",
            ])
        
        if 'best_step' in metrics and metrics['best_step']:
            report_lines.extend([
                "",
                f"- **最佳模型步数**: {metrics['best_step']}",
            ])
    
    report_file = output_dir / 'training_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 训练报告已保存: {report_file}")
    
    # 同时保存JSON格式的指标
    json_file = output_dir / 'training_metrics.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 训练指标JSON已保存: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="分析训练日志并生成报告",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="训练日志文件路径（支持通配符，如 logs/train_*.log）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认: 日志文件所在目录的analysis子目录）"
    )
    
    args = parser.parse_args()
    
    # 处理通配符
    log_files = sorted(Path('.').glob(args.log_file))
    
    if not log_files:
        print(f"错误: 未找到匹配的日志文件: {args.log_file}")
        return 1
    
    # 使用最新的日志文件
    log_file = log_files[-1]
    print(f"使用日志文件: {log_file}")
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = log_file.parent / 'analysis'
    
    print(f"输出目录: {output_dir}")
    print()
    
    # 解析日志
    print("【步骤1】解析训练日志...")
    metrics = parse_log_file(log_file)
    
    if not metrics['step']:
        print("⚠️  警告: 未能从日志中提取到训练指标")
        print("   请检查日志文件格式是否正确")
        return 1
    
    print(f"✓ 解析完成:")
    print(f"  - 训练步数: {len(metrics['step'])}")
    print(f"  - 验证次数: {len(metrics['val_loss'])}")
    print()
    
    # 绘制曲线
    print("【步骤2】绘制训练曲线...")
    plot_training_curves(metrics, output_dir)
    print()
    
    # 生成报告
    print("【步骤3】生成训练报告...")
    generate_report(metrics, log_file, output_dir)
    print()
    
    print("=" * 60)
    print("✓ 分析完成！")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
