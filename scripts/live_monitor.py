#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时可视化实验监控脚本
动态显示实验进度、结果和觉醒时刻
"""

import os
import sys
import time
import re
from datetime import datetime
from collections import deque

# ANSI颜色代码
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    
def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')

def format_time(seconds):
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def draw_progress_bar(current, total, width=50):
    """绘制进度条"""
    if total == 0:
        return " " * width
    percentage = current / total
    filled = int(width * percentage)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage*100:.1f}%"

def get_latest_progress(log_file):
    """从日志中提取最新进度"""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # 从后往前查找最新进度
            for line in reversed(lines[-1000:]):  # 只检查最后1000行
                # 提取情绪向量进度 - 格式: "提取情绪向量（并发）:  53%|█████▎    | 5272/10000 [1:22:55<1:20:06,  1.02s/it]"
                match = re.search(r'提取情绪向量.*?(\d+)/(\d+).*?\[(\d+):(\d+):(\d+)', line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    h, m, s = map(int, match.groups()[2:])
                    # 将运行时间转换为秒（从开始到现在）
                    timestamp = h * 3600 + m * 60 + s
                    return current, total, timestamp
                # 简化匹配（如果没有时间戳）
                match = re.search(r'提取情绪向量.*?(\d+)/(\d+)', line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    return current, total, None
                # 检查是否完成
                if '✅' in line and ('完成' in line or 'finish' in line.lower()):
                    return total, total, None
    except Exception as e:
        pass
    return None, None, None

def get_progress_samples(log_file, max_samples=20):
    """从日志中提取多个进度样本用于计算速度"""
    samples = []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in reversed(lines[-5000:]):  # 检查最后5000行
                # 提取情绪向量进度和时间戳 - 格式: "[1:22:55<1:20:06,  1.02s/it]"
                match = re.search(r'提取情绪向量.*?(\d+)/(\d+).*?\[(\d+):(\d+):(\d+)', line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    h, m, s = map(int, match.groups()[2:])
                    timestamp = h * 3600 + m * 60 + s
                    
                    samples.append((current, total, timestamp))
                    if len(samples) >= max_samples:
                        break
    except Exception as e:
        pass
    
    # 按时间戳排序（从早到晚）
    samples.sort(key=lambda x: x[2] if x[2] is not None else 0)
    return samples

def get_recall_results(log_file):
    """从日志中提取Recall@1结果"""
    results = {}
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # 查找HippoRAG结果
            hippo_match = re.search(r'HippoRAG.*?Recall@1[:\s]+([\d.]+)', content)
            if hippo_match:
                results['hipporag'] = float(hippo_match.group(1))
            
            # 查找HyperAmy结果
            hyper_match = re.search(r'HyperAmy.*?Recall@1[:\s]+([\d.]+)', content, re.IGNORECASE)
            if hyper_match:
                results['hyperamy'] = float(hyper_match.group(1))
            
            # 查找Hybrid结果
            hybrid_match = re.search(r'Hybrid.*?Recall@1[:\s]+([\d.]+)', content, re.IGNORECASE)
            if hybrid_match:
                results['hybrid'] = float(hybrid_match.group(1))
            
            # 也查找通用格式
            for line in content.split('\n')[-500:]:
                if 'Recall@1' in line:
                    match = re.search(r'Recall@1[:\s]+([\d.]+)', line)
                    if match:
                        value = float(match.group(1))
                        if 'HippoRAG' in line or 'hipporag' in line.lower():
                            results['hipporag'] = value
                        elif 'HyperAmy' in line or 'hyperamy' in line.lower():
                            results['hyperamy'] = value
                        elif 'Hybrid' in line or 'hybrid' in line.lower():
                            results['hybrid'] = value
    except Exception as e:
        pass
    return results

def get_eureka_stats(eureka_file):
    """统计觉醒时刻"""
    stats = {
        'total': 0,
        'extreme': 0,
        'w_emo_values': [],
        'latest': None
    }
    try:
        with open(eureka_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                if '觉醒时刻捕获' in line or 'Dynamic Weighting' in line:
                    stats['total'] += 1
                    # 提取W_emo值
                    w_emo_match = re.search(r'W_emo=([\d.]+)', line)
                    if w_emo_match:
                        w_emo = float(w_emo_match.group(1))
                        stats['w_emo_values'].append(w_emo)
                        if w_emo >= 0.7:
                            stats['extreme'] += 1
                    stats['latest'] = line.strip()
    except Exception as e:
        pass
    return stats

def get_process_info(pid):
    """获取进程信息"""
    try:
        import subprocess
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'etime=', 'pcpu=', 'pmem='], 
                              capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) >= 3:
                return {
                    'time': parts[0],
                    'cpu': parts[1],
                    'mem': parts[2]
                }
    except:
        pass
    return None

def main():
    log_file = 'test_vibe_search_experiment_run.log'
    eureka_file = 'eureka_moments.log'
    experiment_pid = '47363'
    monitor_pid = '63356'
    
    # 存储历史数据用于趋势显示
    progress_history = deque(maxlen=20)
    w_emo_history = deque(maxlen=20)
    
    print(f"{Colors.CYAN}{Colors.BOLD}🚀 实时实验监控系统{Colors.RESET}")
    print(f"日志文件: {log_file}")
    print(f"觉醒时刻: {eureka_file}")
    print(f"{Colors.YELLOW}按 Ctrl+C 退出{Colors.RESET}\n")
    
    try:
        while True:
            clear_screen()
            
            # 头部信息
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{Colors.CYAN}{Colors.BOLD}═══════════════════════════════════════════════════════{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}🚀 Vibe Search 实验实时监控{Colors.RESET}")
            print(f"{Colors.CYAN}时间: {now}{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}═══════════════════════════════════════════════════════{Colors.RESET}\n")
            
            # 1. 实验进程状态
            print(f"{Colors.BOLD}📊 实验进程状态{Colors.RESET}")
            process_info = get_process_info(experiment_pid)
            if process_info:
                print(f"  PID: {Colors.GREEN}{experiment_pid}{Colors.RESET} ✅ 运行中")
                print(f"  运行时间: {Colors.YELLOW}{process_info['time']}{Colors.RESET}")
                print(f"  CPU: {Colors.YELLOW}{process_info['cpu']}%{Colors.RESET} | MEM: {Colors.YELLOW}{process_info['mem']}%{Colors.RESET}")
            else:
                print(f"  PID: {Colors.RED}{experiment_pid}{Colors.RESET} ❌ 进程不存在（可能已完成）")
            print()
            
            # 2. 当前进度
            print(f"{Colors.BOLD}📈 HyperAmy 索引进度{Colors.RESET}")
            current, total, timestamp = get_latest_progress(log_file)
            if current is not None and total is not None:
                progress_bar = draw_progress_bar(current, total, width=40)
                print(f"  {progress_bar}")
                print(f"  {Colors.CYAN}{current:,} / {total:,}{Colors.RESET}")
                
                # 计算剩余时间（基于历史速度样本）
                samples = get_progress_samples(log_file, max_samples=10)
                if len(samples) >= 2:
                    # 使用最近的样本计算速度
                    recent_samples = samples[-5:] if len(samples) >= 5 else samples
                    speeds = []
                    
                    for i in range(1, len(recent_samples)):
                        curr_idx, curr_total, curr_time = recent_samples[i]
                        prev_idx, prev_total, prev_time = recent_samples[i-1]
                        
                        if curr_time is not None and prev_time is not None and curr_time > prev_time:
                            items = curr_idx - prev_idx
                            time_delta = curr_time - prev_time
                            if time_delta > 0 and items > 0:
                                speed = items / time_delta
                                speeds.append(speed)
                    
                    if speeds:
                        # 使用最近速度的平均值（更权重最近的速度）
                        if len(speeds) >= 3:
                            # 加权平均：最近的速度权重更大（从后往前）
                            speed_list = list(reversed(speeds))  # 最新的在前面
                            if len(speed_list) >= 5:
                                weights = [0.35, 0.25, 0.20, 0.12, 0.08][:len(speed_list)]
                            elif len(speed_list) == 4:
                                weights = [0.40, 0.30, 0.20, 0.10]
                            else:  # len == 3
                                weights = [0.50, 0.30, 0.20]
                            # 归一化权重
                            total_weight = sum(weights)
                            weights = [w / total_weight for w in weights]
                            avg_speed = sum(s * w for s, w in zip(speed_list, weights))
                        else:
                            avg_speed = sum(speeds) / len(speeds)
                        
                        remaining = total - current
                        if avg_speed > 0 and remaining > 0:
                            eta_seconds = remaining / avg_speed
                            eta_str = format_time(eta_seconds)
                            speed_str = f"{avg_speed:.2f}"
                            print(f"  当前速度: {Colors.CYAN}{speed_str} it/s{Colors.RESET}")
                            print(f"  预计剩余: {Colors.YELLOW}{eta_str}{Colors.RESET}")
                elif current < total:
                    # 如果无法计算精确ETA，使用粗略估计
                    print(f"  {Colors.YELLOW}计算ETA中...{Colors.RESET}")
            else:
                print(f"  {Colors.YELLOW}等待进度更新...{Colors.RESET}")
            print()
            
            # 3. Recall@1 结果
            print(f"{Colors.BOLD}🎯 Recall@1 结果对比{Colors.RESET}")
            results = get_recall_results(log_file)
            
            # HippoRAG
            hippo_recall = results.get('hipporag', None)
            if hippo_recall is not None:
                print(f"  {Colors.RED}HippoRAG{Colors.RESET}: {Colors.RED}{hippo_recall*100:.1f}%{Colors.RESET} ✅ (已确认)")
            else:
                print(f"  {Colors.RED}HippoRAG{Colors.RESET}: {Colors.YELLOW}等待结果...{Colors.RESET}")
            
            # HyperAmy
            hyper_recall = results.get('hyperamy', None)
            if hyper_recall is not None:
                color = Colors.GREEN if hyper_recall >= 0.7 else Colors.YELLOW
                print(f"  {Colors.MAGENTA}HyperAmy{Colors.RESET}: {color}{hyper_recall*100:.1f}%{Colors.RESET}")
            else:
                print(f"  {Colors.MAGENTA}HyperAmy{Colors.RESET}: {Colors.YELLOW}等待结果...{Colors.RESET}")
            
            # Hybrid
            hybrid_recall = results.get('hybrid', None)
            if hybrid_recall is not None:
                color = Colors.GREEN if hybrid_recall >= 0.8 else Colors.YELLOW
                print(f"  {Colors.BLUE}Hybrid (Dynamic v2){Colors.RESET}: {color}{hybrid_recall*100:.1f}%{Colors.RESET}")
            else:
                print(f"  {Colors.BLUE}Hybrid (Dynamic v2){Colors.RESET}: {Colors.YELLOW}等待结果...{Colors.RESET}")
            print()
            
            # 4. 觉醒时刻统计
            print(f"{Colors.BOLD}⚡ 觉醒时刻统计 (Eureka Moments){Colors.RESET}")
            eureka_stats = get_eureka_stats(eureka_file)
            print(f"  总觉醒时刻 (W_emo >= 0.3): {Colors.GREEN}{eureka_stats['total']}{Colors.RESET}")
            print(f"  极端觉醒 (W_emo >= 0.7): {Colors.MAGENTA}{eureka_stats['extreme']}{Colors.RESET} ⚡⚡⚡")
            
            if eureka_stats['w_emo_values']:
                avg_w_emo = sum(eureka_stats['w_emo_values']) / len(eureka_stats['w_emo_values'])
                max_w_emo = max(eureka_stats['w_emo_values'])
                min_w_emo = min(eureka_stats['w_emo_values'])
                print(f"  平均 W_emo: {Colors.CYAN}{avg_w_emo:.3f}{Colors.RESET}")
                print(f"  最大 W_emo: {Colors.MAGENTA}{max_w_emo:.3f}{Colors.RESET}")
                print(f"  最小 W_emo: {Colors.YELLOW}{min_w_emo:.3f}{Colors.RESET}")
                
                if eureka_stats['latest']:
                    print(f"  最新觉醒: {Colors.YELLOW}{eureka_stats['latest'][:80]}...{Colors.RESET}")
            else:
                print(f"  {Colors.YELLOW}等待第一个觉醒时刻...{Colors.RESET}")
            print()
            
            # 5. 监控进程状态
            print(f"{Colors.BOLD}👁️  监控系统状态{Colors.RESET}")
            monitor_info = get_process_info(monitor_pid)
            if monitor_info:
                print(f"  监控PID: {Colors.GREEN}{monitor_pid}{Colors.RESET} ✅ 监控中")
            else:
                print(f"  监控PID: {Colors.YELLOW}{monitor_pid}{Colors.RESET} ⚠️  状态未知")
            print()
            
            # 6. 实验阶段判断
            print(f"{Colors.BOLD}📋 当前实验阶段{Colors.RESET}")
            if current is not None and total is not None:
                if current < total:
                    stage = "HyperAmy 索引中 (提取情绪向量)"
                    print(f"  {Colors.CYAN}{stage}{Colors.RESET}")
                else:
                    stage = "索引完成，进入检索阶段"
                    print(f"  {Colors.GREEN}{stage}{Colors.RESET}")
            else:
                print(f"  {Colors.YELLOW}等待状态更新...{Colors.RESET}")
            
            # 底部提示
            print(f"\n{Colors.CYAN}{Colors.BOLD}═══════════════════════════════════════════════════════{Colors.RESET}")
            print(f"{Colors.YELLOW}每2秒自动刷新 | 按 Ctrl+C 退出{Colors.RESET}")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}{Colors.BOLD}✅ 监控已停止{Colors.RESET}")
        print(f"{Colors.CYAN}实验仍在后台运行中{Colors.RESET}")

if __name__ == '__main__':
    main()
