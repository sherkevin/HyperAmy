#!/usr/bin/env python3
"""
绘制二阶段训练Loss曲线
"""

import json
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 配置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(losses: list, output_file: Path):
    """
    绘制训练Loss曲线
    
    Args:
        losses: Loss值列表，例如 [0.0998, 0.0848, 0.0707, 0.0632, 0.0527]
        output_file: 输出文件路径
    """
    epochs = list(range(1, len(losses) + 1))
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制Loss曲线
    ax.plot(epochs, losses, marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Training Loss')
    
    # 添加数值标签
    for i, (epoch, loss) in enumerate(zip(epochs, losses)):
        ax.annotate(
            f'{loss:.4f}',
            xy=(epoch, loss),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            color='#2E86AB'
        )
    
    # 设置标签和标题
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Stage 2 Training: Contrastive Learning Loss Curve', fontsize=14, fontweight='bold')
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 设置x轴刻度
    ax.set_xticks(epochs)
    ax.set_xlim(0.5, len(epochs) + 0.5)
    
    # 设置y轴范围（留出一些空间）
    y_min = min(losses) * 0.95
    y_max = max(losses) * 1.05
    ax.set_ylim(y_min, y_max)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=11)
    
    # 添加统计信息
    loss_reduction = ((losses[0] - losses[-1]) / losses[0]) * 100
    info_text = f'Loss Reduction: {loss_reduction:.1f}%\nInitial Loss: {losses[0]:.4f}\nFinal Loss: {losses[-1]:.4f}'
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # 同时保存PDF版本
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    
    print(f"✅ Loss曲线已保存:")
    print(f"   PNG: {output_file}")
    print(f"   PDF: {pdf_file}")


def main():
    parser = argparse.ArgumentParser(description="绘制二阶段训练Loss曲线")
    parser.add_argument(
        "--losses",
        type=str,
        default="0.0998,0.0848,0.0707,0.0632,0.0527",
        help="Loss值列表（逗号分隔）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="docs/figures/stage2_training_loss_curves.png",
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    # 解析Loss值
    try:
        losses = [float(x.strip()) for x in args.losses.split(',')]
    except ValueError:
        print(f"❌ 无法解析Loss值: {args.losses}")
        print("   格式应为: 0.0998,0.0848,0.0707,0.0632,0.0527")
        return
    
    # 创建输出目录
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 绘制曲线
    plot_training_curves(losses, output_file)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("训练Loss统计")
    print("=" * 60)
    print(f"Epochs: {len(losses)}")
    print(f"Initial Loss: {losses[0]:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Loss Reduction: {((losses[0] - losses[-1]) / losses[0]) * 100:.1f}%")
    print(f"Average Loss: {sum(losses) / len(losses):.4f}")


if __name__ == "__main__":
    main()
