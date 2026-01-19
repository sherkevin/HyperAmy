#!/usr/bin/env python3
"""
自动从训练日志中提取实际训练速度和统计数据
用于完善实验记录表
"""

import re
import sys
from datetime import datetime
from pathlib import Path
import json

def extract_training_stats(log_file):
    """从日志文件中提取训练统计数据"""
    results = {
        'log_file': str(log_file),
        'steps': [],
        'epochs': {},
        'stats': {}
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"无法读取日志文件: {e}")
        return None
    
    fmt = "%Y-%m-%d %H:%M:%S"
    
    # 提取所有step的时间戳
    for line in lines:
        # 提取Step信息: "2026-01-15 08:12:50 - ... Step 99: ..."
        match = re.search(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*Step (\d+):', line)
        if match:
            step_num = int(match.group(3))
            step_time = f"{match.group(1)} {match.group(2)}"
            try:
                timestamp = datetime.strptime(step_time, fmt).timestamp()
                results['steps'].append({
                    'step': step_num,
                    'time': step_time,
                    'timestamp': timestamp
                })
            except:
                pass
    
    # 提取Epoch信息
    for line in lines:
        match = re.search(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*Starting Epoch (\d+)/', line)
        if match:
            epoch_num = int(match.group(3))
            epoch_time = f"{match.group(1)} {match.group(2)}"
            if epoch_num not in results['epochs']:
                results['epochs'][epoch_num] = {'start': epoch_time}
        
        # 查找validation结果作为epoch结束
        match = re.search(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*Epoch (\d+)/.*Val Loss', line)
        if match:
            epoch_num = int(match.group(3))
            epoch_time = f"{match.group(1)} {match.group(2)}"
            if epoch_num in results['epochs']:
                results['epochs'][epoch_num]['end'] = epoch_time
    
    # 计算统计信息
    if len(results['steps']) >= 2:
        first = results['steps'][0]
        last = results['steps'][-1]
        time_diff = last['timestamp'] - first['timestamp']
        step_diff = last['step'] - first['step']
        
        if step_diff > 0:
            sec_per_step = time_diff / step_diff
            
            # 假设每个epoch 1609步（可从config获取）
            steps_per_epoch = 1609
            sec_per_epoch = sec_per_step * steps_per_epoch
            
            completed_epochs = last['step'] // steps_per_epoch
            remaining_steps_in_epoch = last['step'] % steps_per_epoch
            
            results['stats'] = {
                'first_step': first['step'],
                'first_step_time': first['time'],
                'last_step': last['step'],
                'last_step_time': last['time'],
                'total_steps': step_diff,
                'total_time_seconds': time_diff,
                'total_time_minutes': time_diff / 60,
                'total_time_hours': time_diff / 3600,
                'seconds_per_step': sec_per_step,
                'minutes_per_epoch': sec_per_epoch / 60,
                'hours_per_epoch': sec_per_epoch / 3600,
                'completed_epochs': completed_epochs,
                'remaining_steps_in_current_epoch': remaining_steps_in_epoch,
                'steps_per_epoch': steps_per_epoch
            }
    
    return results


def print_stats(results):
    """打印统计信息"""
    if not results or 'stats' not in results or not results['stats']:
        print("无法提取统计数据")
        return
    
    stats = results['stats']
    print("=" * 60)
    print("训练速度统计")
    print("=" * 60)
    print(f"日志文件: {results['log_file']}")
    print()
    print("时间范围:")
    print(f"  开始: Step {stats['first_step']} at {stats['first_step_time']}")
    print(f"  结束: Step {stats['last_step']} at {stats['last_step_time']}")
    print()
    print("总统计:")
    print(f"  总步数: {stats['total_steps']:,}步")
    print(f"  总时间: {stats['total_time_hours']:.2f}小时 ({stats['total_time_minutes']:.1f}分钟)")
    print()
    print("速度指标:")
    print(f"  每个step耗时: {stats['seconds_per_step']:.2f}秒")
    print(f"  每个epoch耗时: {stats['minutes_per_epoch']:.1f}分钟 ({stats['hours_per_epoch']:.3f}小时)")
    print()
    print("进度:")
    print(f"  已完成epoch数: {stats['completed_epochs']}")
    print(f"  当前epoch进度: {stats['remaining_steps_in_current_epoch']}/{stats['steps_per_epoch']}")
    
    # 估算剩余时间
    if stats['completed_epochs'] < 20:  # 假设总共20个epoch
        remaining_epochs = 20 - stats['completed_epochs']
        if stats['remaining_steps_in_current_epoch'] > 0:
            remaining_steps = stats['steps_per_epoch'] - stats['remaining_steps_in_current_epoch']
            remaining_time = remaining_steps * stats['seconds_per_step'] + (remaining_epochs - 1) * stats['seconds_per_epoch']
        else:
            remaining_time = remaining_epochs * stats['seconds_per_epoch']
        
        print()
        print("剩余时间估算:")
        print(f"  剩余epoch数: {remaining_epochs}")
        print(f"  剩余时间: {remaining_time/60:.1f}分钟 ({remaining_time/3600:.2f}小时)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python extract_training_stats.py <log_file>")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"日志文件不存在: {log_file}")
        sys.exit(1)
    
    results = extract_training_stats(log_file)
    if results:
        print_stats(results)
        
        # 保存为JSON
        json_file = log_file.parent / f"{log_file.stem}_stats.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print()
        print(f"统计信息已保存到: {json_file}")
