#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制两个模型的训练Loss曲线对比
"""
import re
import sys
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_training_log(log_file):
    """解析训练日志，提取loss信息"""
    epochs = []
    train_losses = []
    val_losses = []
    val_vmf_losses = []
    val_cal_losses = []
    val_aux_losses = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_epoch = None
    
    for line in lines:
        # 匹配epoch开始
        epoch_match = re.search(r'Epoch (\d+)/\d+', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # 匹配训练loss
        train_match = re.search(r'Train Loss: ([\d.]+)', line)
        if train_match and current_epoch:
            train_loss = float(train_match.group(1))
            if current_epoch not in [e['epoch'] for e in epochs]:
                epochs.append({'epoch': current_epoch})
            idx = len(epochs) - 1
            train_losses.append(train_loss)
        
        # 匹配验证loss
        val_match = re.search(r'Val Loss: ([\d.]+)', line)
        if val_match and current_epoch:
            val_loss = float(val_match.group(1))
            val_losses.append(val_loss)
            
            # 提取vMF, Cal, Aux loss
            vmf_match = re.search(r'vmf=([\d.]+)', line)
            cal_match = re.search(r'cal=([\d.]+)', line)
            aux_match = re.search(r'aux=([\d.]+)', line)
            
            if vmf_match:
                val_vmf_losses.append(float(vmf_match.group(1)))
            if cal_match:
                val_cal_losses.append(float(cal_match.group(1)))
            if aux_match:
                val_aux_losses.append(float(aux_match.group(1)))
    
    return {
        'epochs': [e['epoch'] for e in epochs],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_vmf_losses': val_vmf_losses,
        'val_cal_losses': val_cal_losses,
        'val_aux_losses': val_aux_losses,
    }

def plot_comparison(qwen_log, roberta_log, output_dir='docs/figures'):
    """绘制两个模型的对比图"""
    
    # 解析日志
    print("解析Qwen-7B训练日志...")
    qwen_data = parse_training_log(qwen_log)
    
    print("解析RoBERTa-large训练日志...")
    roberta_data = parse_training_log(roberta_log)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Training Comparison: Qwen-7B vs RoBERTa-large', fontsize=16, fontweight='bold')
    
    # 1. 验证Loss对比
    ax1 = axes[0, 0]
    if qwen_data['val_losses']:
        ax1.plot(qwen_data['epochs'][:len(qwen_data['val_losses'])], 
                qwen_data['val_losses'], 
                'o-', label='Qwen-7B', linewidth=2, markersize=6, color='#1f77b4')
    if roberta_data['val_losses']:
        ax1.plot(roberta_data['epochs'][:len(roberta_data['val_losses'])], 
                roberta_data['val_losses'], 
                's-', label='RoBERTa-large', linewidth=2, markersize=6, color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. 训练Loss对比
    ax2 = axes[0, 1]
    if qwen_data['train_losses']:
        ax2.plot(qwen_data['epochs'][:len(qwen_data['train_losses'])], 
                qwen_data['train_losses'], 
                'o-', label='Qwen-7B', linewidth=2, markersize=6, color='#1f77b4')
    if roberta_data['train_losses']:
        ax2.plot(roberta_data['epochs'][:len(roberta_data['train_losses'])], 
                roberta_data['train_losses'], 
                's-', label='RoBERTa-large', linewidth=2, markersize=6, color='#ff7f0e')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Loss组件对比（vMF Loss）
    ax3 = axes[1, 0]
    if qwen_data['val_vmf_losses']:
        ax3.plot(range(1, len(qwen_data['val_vmf_losses'])+1), 
                qwen_data['val_vmf_losses'], 
                'o-', label='Qwen-7B', linewidth=2, markersize=6, color='#1f77b4')
    if roberta_data['val_vmf_losses']:
        ax3.plot(range(1, len(roberta_data['val_vmf_losses'])+1), 
                roberta_data['val_vmf_losses'], 
                's-', label='RoBERTa-large', linewidth=2, markersize=6, color='#ff7f0e')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('vMF Loss', fontsize=12)
    ax3.set_title('vMF Loss Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss组件对比（Calibration Loss）
    ax4 = axes[1, 1]
    if qwen_data['val_cal_losses']:
        ax4.plot(range(1, len(qwen_data['val_cal_losses'])+1), 
                qwen_data['val_cal_losses'], 
                'o-', label='Qwen-7B', linewidth=2, markersize=6, color='#1f77b4')
    if roberta_data['val_cal_losses']:
        ax4.plot(range(1, len(roberta_data['val_cal_losses'])+1), 
                roberta_data['val_cal_losses'], 
                's-', label='RoBERTa-large', linewidth=2, markersize=6, color='#ff7f0e')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Calibration Loss', fontsize=12)
    ax4.set_title('Calibration Loss Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(output_dir, 'both_models_loss_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {output_file}")
    
    # 打印统计信息
    print("\n" + "="*70)
    print("训练统计")
    print("="*70)
    
    if qwen_data['val_losses']:
        print(f"\nQwen-7B:")
        print(f"  最佳验证Loss: {min(qwen_data['val_losses']):.4f} (Epoch {qwen_data['val_losses'].index(min(qwen_data['val_losses']))+1})")
        print(f"  最终验证Loss: {qwen_data['val_losses'][-1]:.4f}")
        print(f"  训练轮数: {len(qwen_data['val_losses'])}")
    
    if roberta_data['val_losses']:
        print(f"\nRoBERTa-large:")
        print(f"  最佳验证Loss: {min(roberta_data['val_losses']):.4f} (Epoch {roberta_data['val_losses'].index(min(roberta_data['val_losses']))+1})")
        print(f"  最终验证Loss: {roberta_data['val_losses'][-1]:.4f}")
        print(f"  训练轮数: {len(roberta_data['val_losses'])}")
    
    if qwen_data['val_losses'] and roberta_data['val_losses']:
        print(f"\n对比:")
        qwen_best = min(qwen_data['val_losses'])
        roberta_best = min(roberta_data['val_losses'])
        if qwen_best < roberta_best:
            print(f"  ✅ Qwen-7B表现更好 (Loss低 {((roberta_best - qwen_best) / roberta_best * 100):.2f}%)")
        else:
            print(f"  ✅ RoBERTa-large表现更好 (Loss低 {((qwen_best - roberta_best) / qwen_best * 100):.2f}%)")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python plot_both_loss_curves.py <qwen_log> <roberta_log>")
        sys.exit(1)
    
    qwen_log = sys.argv[1]
    roberta_log = sys.argv[2]
    
    if not os.path.exists(qwen_log):
        print(f"错误: Qwen日志文件不存在: {qwen_log}")
        sys.exit(1)
    
    if not os.path.exists(roberta_log):
        print(f"错误: RoBERTa日志文件不存在: {roberta_log}")
        sys.exit(1)
    
    plot_comparison(qwen_log, roberta_log)
