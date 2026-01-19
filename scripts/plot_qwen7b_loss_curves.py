#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制Qwen-7B模型训练Loss曲线
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
    train_vmf_losses = []
    train_cal_losses = []
    train_aux_losses = []
    
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
            train_losses.append(train_loss)
            
            # 提取训练loss组件
            vmf_match = re.search(r'vmf=([\d.]+)', line)
            cal_match = re.search(r'cal=([\d.]+)', line)
            aux_match = re.search(r'aux=([\d.]+)', line)
            
            if vmf_match:
                train_vmf_losses.append(float(vmf_match.group(1)))
            if cal_match:
                train_cal_losses.append(float(cal_match.group(1)))
            if aux_match:
                train_aux_losses.append(float(aux_match.group(1)))
        
        # 匹配验证loss
        val_match = re.search(r'Val Loss: ([\d.]+)', line)
        if val_match and current_epoch:
            val_loss = float(val_match.group(1))
            val_losses.append(val_loss)
            
            # 提取验证loss组件
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
        'train_vmf_losses': train_vmf_losses,
        'train_cal_losses': train_cal_losses,
        'train_aux_losses': train_aux_losses,
    }

def plot_qwen7b_losses(log_file, output_dir='docs/figures'):
    """绘制Qwen-7B模型的loss曲线"""
    
    print(f"解析训练日志: {log_file}")
    data = parse_training_log(log_file)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Qwen-7B Model Training Loss Curves', fontsize=16, fontweight='bold')
    
    epochs_list = list(range(1, len(data['train_losses']) + 1))
    
    # 1. 总Loss（训练 vs 验证）
    ax1 = axes[0, 0]
    if data['train_losses']:
        ax1.plot(epochs_list, data['train_losses'], 'o-', label='Train Loss', 
                linewidth=2, markersize=6, color='#1f77b4')
    if data['val_losses']:
        ax1.plot(epochs_list[:len(data['val_losses'])], data['val_losses'], 's-', 
                label='Val Loss', linewidth=2, markersize=6, color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Total Loss: Train vs Validation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. vMF Loss
    ax2 = axes[0, 1]
    if data['train_vmf_losses']:
        ax2.plot(epochs_list[:len(data['train_vmf_losses'])], data['train_vmf_losses'], 
                'o-', label='Train vMF', linewidth=2, markersize=6, color='#1f77b4')
    if data['val_vmf_losses']:
        ax2.plot(epochs_list[:len(data['val_vmf_losses'])], data['val_vmf_losses'], 
                's-', label='Val vMF', linewidth=2, markersize=6, color='#ff7f0e')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('vMF Loss', fontsize=12)
    ax2.set_title('vMF Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Calibration Loss
    ax3 = axes[0, 2]
    if data['train_cal_losses']:
        ax3.plot(epochs_list[:len(data['train_cal_losses'])], data['train_cal_losses'], 
                'o-', label='Train Cal', linewidth=2, markersize=6, color='#1f77b4')
    if data['val_cal_losses']:
        ax3.plot(epochs_list[:len(data['val_cal_losses'])], data['val_cal_losses'], 
                's-', label='Val Cal', linewidth=2, markersize=6, color='#ff7f0e')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Calibration Loss', fontsize=12)
    ax3.set_title('Calibration Loss', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Auxiliary Loss
    ax4 = axes[1, 0]
    if data['train_aux_losses']:
        ax4.plot(epochs_list[:len(data['train_aux_losses'])], data['train_aux_losses'], 
                'o-', label='Train Aux', linewidth=2, markersize=6, color='#1f77b4')
    if data['val_aux_losses']:
        ax4.plot(epochs_list[:len(data['val_aux_losses'])], data['val_aux_losses'], 
                's-', label='Val Aux', linewidth=2, markersize=6, color='#ff7f0e')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Auxiliary Loss', fontsize=12)
    ax4.set_title('Auxiliary Loss', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 5. 验证Loss组件对比
    ax5 = axes[1, 1]
    if data['val_vmf_losses']:
        ax5.plot(epochs_list[:len(data['val_vmf_losses'])], data['val_vmf_losses'], 
                'o-', label='vMF', linewidth=2, markersize=6, color='#2ca02c')
    if data['val_cal_losses']:
        ax5.plot(epochs_list[:len(data['val_cal_losses'])], data['val_cal_losses'], 
                's-', label='Cal', linewidth=2, markersize=6, color='#d62728')
    if data['val_aux_losses']:
        ax5.plot(epochs_list[:len(data['val_aux_losses'])], data['val_aux_losses'], 
                '^-', label='Aux', linewidth=2, markersize=6, color='#9467bd')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Validation Loss Components', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. Loss下降趋势（验证）
    ax6 = axes[1, 2]
    if data['val_losses']:
        val_losses_array = np.array(data['val_losses'])
        best_loss = np.min(val_losses_array)
        best_epoch = np.argmin(val_losses_array) + 1
        ax6.plot(epochs_list[:len(data['val_losses'])], val_losses_array, 
                'o-', linewidth=2, markersize=6, color='#ff7f0e')
        ax6.axhline(y=best_loss, color='r', linestyle='--', linewidth=1.5, 
                   label=f'Best: {best_loss:.4f} (Epoch {best_epoch})')
        ax6.scatter([best_epoch], [best_loss], color='r', s=100, zorder=5)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Validation Loss', fontsize=12)
    ax6.set_title('Validation Loss Trend', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(output_dir, 'qwen7b_training_loss_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {output_file}")
    
    # 打印统计信息
    print("\n" + "="*70)
    print("Qwen-7B 训练统计")
    print("="*70)
    
    if data['val_losses']:
        best_val_loss = min(data['val_losses'])
        best_epoch = data['val_losses'].index(best_val_loss) + 1
        print(f"\n最佳验证Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
        print(f"最终验证Loss: {data['val_losses'][-1]:.4f}")
        print(f"训练轮数: {len(data['val_losses'])} epochs")
        
        if data['val_vmf_losses']:
            print(f"\n验证Loss组件:")
            print(f"  vMF Loss: {data['val_vmf_losses'][-1]:.4f} (最佳: {min(data['val_vmf_losses']):.4f})")
        if data['val_cal_losses']:
            print(f"  Cal Loss: {data['val_cal_losses'][-1]:.4f} (最佳: {min(data['val_cal_losses']):.4f})")
        if data['val_aux_losses']:
            print(f"  Aux Loss: {data['val_aux_losses'][-1]:.4f} (最佳: {min(data['val_aux_losses']):.4f})")
    
    if data['train_losses']:
        print(f"\n训练Loss: {data['train_losses'][-1]:.4f} (初始: {data['train_losses'][0]:.4f})")
        print(f"Loss下降: {data['train_losses'][0] - data['train_losses'][-1]:.4f} ({((data['train_losses'][0] - data['train_losses'][-1]) / data['train_losses'][0] * 100):.2f}%)")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认从服务器获取日志
        print("从服务器获取训练日志...")
        import subprocess
        result = subprocess.run(
            ['ssh', '-p', '1066', 'jiangh@10.103.92.120', 
             'ls -t /public/jiangh/emos/logs/train_qwen7b_optimized_*.log | head -1'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            log_file = result.stdout.strip()
            print(f"找到日志文件: {log_file}")
            # 下载日志到本地
            local_log = '/tmp/qwen7b_training.log'
            subprocess.run(['scp', '-P', '1066', f'jiangh@10.103.92.120:{log_file}', local_log])
            log_file = local_log
        else:
            print("错误: 无法找到训练日志，请手动指定日志文件路径")
            print("用法: python plot_qwen7b_loss_curves.py <log_file>")
            sys.exit(1)
    else:
        log_file = sys.argv[1]
    
    if not os.path.exists(log_file):
        print(f"错误: 日志文件不存在: {log_file}")
        sys.exit(1)
    
    plot_qwen7b_losses(log_file)
