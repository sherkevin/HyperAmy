#!/usr/bin/env python3
"""
绘制emos训练loss曲线

从训练日志中提取loss数据并绘制曲线图
"""

import argparse
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def parse_training_log(log_file: Path):
    """解析训练日志，提取loss数据"""
    epochs = []
    train_losses = []
    val_losses = []
    kappas = []
    
    current_epoch = 0
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配epoch信息
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # 匹配训练loss
            train_match = re.search(r'Train Loss: ([\d.]+)', line)
            if train_match:
                train_loss = float(train_match.group(1))
                epochs.append(current_epoch)
                train_losses.append(train_loss)
            
            # 匹配验证loss
            val_match = re.search(r'Val Loss: ([\d.]+)', line)
            if val_match:
                val_loss = float(val_match.group(1))
                val_losses.append(val_loss)
            
            # 匹配kappa
            kappa_match = re.search(r'Avg Kappa: ([\d.]+)', line)
            if kappa_match:
                kappa = float(kappa_match.group(1))
                kappas.append(kappa)
    
    return epochs, train_losses, val_losses, kappas


def plot_training_curves(epochs, train_losses, val_losses, kappas, output_file: Path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Loss曲线
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(epochs, val_losses, 's-', label='Val Loss', linewidth=2, markersize=8, color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='best', labels=['Train Loss', 'Val Loss'])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(epochs)
    
    # 标注最佳模型
    if val_losses:
        best_idx = val_losses.index(min(val_losses))
        best_epoch = epochs[best_idx]
        best_val_loss = val_losses[best_idx]
        ax1.annotate(f'Best Model\nEpoch {best_epoch}\nVal Loss: {best_val_loss:.2f}',
                    xy=(best_epoch, best_val_loss),
                    xytext=(best_epoch + 1, best_val_loss + max(val_losses) * 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    ha='left')
    
    # Kappa曲线
    if kappas:
        ax2 = axes[1]
        ax2.plot(epochs, kappas, 'o-', label='Avg Kappa', linewidth=2, markersize=8, color='#F18F01')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Kappa', fontsize=12, fontweight='bold')
        ax2.set_title('Average Kappa Curve', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Loss曲线图已保存: {output_file}")
    
    # 同时保存PDF版本
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ Loss曲线图（PDF）已保存: {pdf_file}")


def main():
    parser = argparse.ArgumentParser(description="绘制emos训练loss曲线")
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="训练日志文件路径（默认：自动查找最新的train_full_*.log）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认：logs目录）"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定日志文件，尝试从服务器获取
    if not args.log_file:
        print("错误: 请指定训练日志文件路径")
        print("使用方法: python scripts/plot_emos_training_curves.py --log_file <日志文件路径>")
        return 1
    
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"错误: 日志文件不存在: {log_file}")
        return 1
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = log_file.parent / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("绘制emos训练Loss曲线")
    print("=" * 70)
    print(f"日志文件: {log_file}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 解析日志
    print("【步骤1】解析训练日志...")
    epochs, train_losses, val_losses, kappas = parse_training_log(log_file)
    
    if not epochs:
        print("错误: 未能从日志中提取到训练数据")
        return 1
    
    print(f"✓ 解析完成: {len(epochs)} 个epochs")
    print()
    
    # 绘制曲线
    print("【步骤2】绘制Loss曲线...")
    output_file = output_dir / 'training_loss_curves.png'
    plot_training_curves(epochs, train_losses, val_losses, kappas, output_file)
    print()
    
    print("=" * 70)
    print("✓ 完成！")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
