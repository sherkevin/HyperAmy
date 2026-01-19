#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三种方法对比实验 - 实时进度监控
支持动态刷新显示实验进度、速度、预计完成时间等
"""
import subprocess
import time
import re
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_process_info(server="hyperamy-server"):
    """获取进程信息"""
    cmd = f'ssh {server} "ps aux | grep \'[t]est_three_methods_comparison_monte_cristo.py\' | grep python"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            if len(parts) >= 11:
                return {
                    'running': True,
                    'pid': parts[1],
                    'cpu': parts[2],
                    'mem': parts[3],
                    'time': parts[9]
                }
    except:
        pass
    return {'running': False}

def get_latest_progress(server="hyperamy-server"):
    """获取最新进度"""
    cmd = f'ssh {server} "cd /public/jiangh/HyperAmy && tail -100 test_three_methods_comparison_monte_cristo.log 2>/dev/null"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                # 支持多种进度条格式
                if ('Extracting emotion vectors:' in line or 'NER:' in line or 'Extracting triples:' in line or 'Processing' in line or 'Batch Encoding' in line) and '%|' in line:
                    # 解析进度条: 72%|███████▏  | 7058/9735 [59:21<34:45,  1.28it/s]
                    match = re.search(r'(\d+)%\|.*?(\d+)/(\d+).*?\[(.*?)<(.*?), (.*?)\]', line)
                    if match:
                        percent = int(match.group(1))
                        current = int(match.group(2))
                        total = int(match.group(3))
                        elapsed = match.group(4)
                        remaining = match.group(5)
                        speed = match.group(6)
                        # 提取任务名称
                        task_name = "未知任务"
                        if 'Extracting emotion vectors' in line:
                            task_name = "提取情绪向量"
                        elif 'NER:' in line:
                            task_name = "命名实体识别 (NER)"
                        elif 'Extracting triples' in line:
                            task_name = "提取三元组 (Triples)"
                        elif 'Processing' in line:
                            task_name = "处理文档"
                        elif 'Batch Encoding' in line:
                            task_name = "批量编码"
                        return {
                            'percent': percent,
                            'current': current,
                            'total': total,
                            'elapsed': elapsed,
                            'remaining': remaining,
                            'speed': speed,
                            'task_name': task_name,
                            'found': True
                        }
    except:
        pass
    return {'found': False}

def get_steps_status(server="hyperamy-server"):
    """获取步骤状态"""
    cmd = f'ssh {server} "cd /public/jiangh/HyperAmy && grep -E \'【步骤|初始化成功|完成|索引完成|检索完成\' test_three_methods_comparison_monte_cristo.log 2>/dev/null | tail -20"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            steps = {
                '步骤1': False,  # 加载数据集
                '步骤2': False,  # HippoRAG初始化
                '步骤3': False,  # HippoRAG索引
                '步骤4': False,  # HippoRAG检索
                '步骤5': False,  # Fusion初始化
                '步骤6': False,  # Fusion索引中
                '步骤7': False,  # Fusion检索
                '步骤8': False,  # HyperAmy初始化
                '步骤9': False,  # HyperAmy索引+检索
                '步骤10': False  # 结果对比
            }
            
            content = result.stdout
            if '步骤1' in content or '加载了' in content:
                steps['步骤1'] = True
            if '步骤2' in content or 'HippoRAG 初始化成功' in content:
                steps['步骤2'] = True
            if '步骤3' in content or 'HippoRAG 索引完成' in content:
                steps['步骤3'] = True
            if '步骤4' in content or 'HippoRAG 检索完成' in content:
                steps['步骤4'] = True
            if '步骤5' in content or 'Fusion 初始化成功' in content:
                steps['步骤5'] = True
            if '步骤6' in content:
                steps['步骤6'] = True  # 正在进行
            if '步骤7' in content or 'Fusion 检索完成' in content:
                steps['步骤7'] = True
            if '步骤8' in content or 'HyperAmy 存储初始化完成' in content:
                steps['步骤8'] = True
            if '步骤9' in content or 'HyperAmy 检索完成' in content:
                steps['步骤9'] = True
            if '步骤10' in content or '结果对比完成' in content:
                steps['步骤10'] = True
            
            return steps
    except:
        pass
    return {}

def get_result_file_status(server="hyperamy-server"):
    """检查结果文件状态"""
    cmd = f'ssh {server} "cd /public/jiangh/HyperAmy && test -f outputs/three_methods_comparison_monte_cristo/comparison_results.json && echo exists || echo not_exists"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if 'exists' in result.stdout:
            return True
    except:
        pass
    return False

def format_time(time_str):
    """格式化时间字符串"""
    return time_str.strip()

def create_progress_bar(percent, width=50):
    """创建进度条"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '▏' * (1 if filled < width and percent > 0 else 0) + '░' * (width - filled - 1)
    return bar

def display_dashboard(process_info, progress, steps, result_exists, update_count=0):
    """显示实时仪表板"""
    clear_screen()
    
    print("=" * 80)
    print("📊 三种方法对比实验 - 实时进度监控")
    print("=" * 80)
    print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 刷新次数: {update_count}")
    print()
    
    # 进程状态
    print("🔄 进程状态:")
    print("-" * 80)
    if process_info.get('running'):
        print(f"  ✅ 运行中 | PID: {process_info['pid']} | CPU: {process_info['cpu']}% | 内存: {process_info['mem']}% | 运行时间: {process_info['time']}")
    else:
        print("  ⏸️  进程未运行")
    print()
    
    # 当前进度
    if progress.get('found'):
        print("📈 当前进度:")
        print("-" * 80)
        percent = progress['percent']
        current = progress['current']
        total = progress['total']
        elapsed = progress['elapsed']
        remaining = progress['remaining']
        speed = progress['speed']
        task_name = progress.get('task_name', '未知任务')
        
        bar = create_progress_bar(percent, width=50)
        print(f"  任务: {task_name}")
        print(f"  进度: {percent}% | {current}/{total}")
        print(f"  [{bar}]")
        print(f"  已用时间: {elapsed} | 预计剩余: {remaining} | 速度: {speed}")
        print()
        
        # 计算完成百分比和详细信息
        remaining_items = total - current
        try:
            # 尝试从速度提取数字
            speed_match = re.search(r'([\d.]+)', speed)
            if speed_match:
                speed_val = float(speed_match.group(1))
                if speed_val > 0:
                    est_seconds = remaining_items / speed_val
                    est_time = timedelta(seconds=int(est_seconds))
                    print(f"  剩余项目: {remaining_items} | 预计完成时间: {est_time}")
        except:
            pass
    else:
        print("📈 当前进度:")
        print("-" * 80)
        print("  ⏳ 等待进度信息...")
    print()
    
    # 步骤状态
    print("📋 实验步骤:")
    print("-" * 80)
    step_names = {
        '步骤1': '加载数据集',
        '步骤2': 'HippoRAG初始化',
        '步骤3': 'HippoRAG索引',
        '步骤4': 'HippoRAG检索',
        '步骤5': 'Fusion初始化',
        '步骤6': 'Fusion索引（情绪提取）',
        '步骤7': 'Fusion检索',
        '步骤8': 'HyperAmy初始化',
        '步骤9': 'HyperAmy索引+检索',
        '步骤10': '结果对比和保存'
    }
    
    for step_key in ['步骤1', '步骤2', '步骤3', '步骤4', '步骤5', '步骤6', '步骤7', '步骤8', '步骤9', '步骤10']:
        status = steps.get(step_key, False)
        step_name = step_names.get(step_key, step_key)
        if status:
            # 检查是否是当前步骤（Fusion索引中）
            if step_key == '步骤6' and progress.get('found') and progress.get('percent', 0) < 100:
                icon = "⏳"  # 进行中
            else:
                icon = "✅"  # 已完成
        else:
            icon = "⏸️ "  # 未开始
        print(f"  {icon} {step_key}: {step_name}")
    print()
    
    # 结果文件状态
    print("📁 结果文件:")
    print("-" * 80)
    if result_exists:
        print("  ✅ 结果文件已生成")
    else:
        print("  ⏳ 结果文件尚未生成")
    print()
    
    # 实时日志（最后几行）
    print("📝 最新日志 (最后3行):")
    print("-" * 80)
    try:
        cmd = f'ssh hyperamy-server "cd /public/jiangh/HyperAmy && tail -3 test_three_methods_comparison_monte_cristo.log 2>/dev/null"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n')[-3:]:
                if line.strip():
                    # 截断过长的行
                    line = line.strip()[:100]
                    print(f"  {line}")
        else:
            print("  ⏳ 等待日志...")
    except:
        print("  ⚠️  无法获取日志")
    print()
    
    print("=" * 80)
    print("💡 提示: 按 Ctrl+C 退出监控")
    print("=" * 80)

def main():
    """主函数"""
    server = "hyperamy-server"
    update_interval = 3  # 每3秒刷新一次
    update_count = 0
    
    print("🚀 启动实时进度监控...")
    print(f"📡 连接服务器: {server}")
    print(f"🔄 刷新间隔: {update_interval}秒")
    print("💡 提示: 按 Ctrl+C 退出监控")
    print()
    time.sleep(2)
    
    try:
        while True:
            update_count += 1
            
            try:
                # 获取所有信息（设置超时以避免卡住）
                process_info = get_process_info(server)
                progress = get_latest_progress(server)
                steps = get_steps_status(server)
                result_exists = get_result_file_status(server)
                
                # 显示仪表板
                display_dashboard(process_info, progress, steps, result_exists, update_count)
                
                # 如果实验完成，检查结果
                if result_exists and not process_info.get('running'):
                    print("\n🎉 实验已完成！结果文件已生成。")
                    print("=" * 80)
                    break
                    
                # 如果进度达到100%，等待一段时间再检查是否完成
                if progress.get('found') and progress.get('percent', 0) >= 100:
                    print("\n⚠️  当前任务已完成，等待实验完成...")
                    time.sleep(10)  # 等待10秒后检查结果
                    result_exists = get_result_file_status(server)
                    if result_exists:
                        print("\n🎉 实验已完成！结果文件已生成。")
                        print("=" * 80)
                        break
                
            except Exception as e:
                # 如果某个操作超时或失败，继续尝试
                print(f"\n⚠️  获取状态时出错: {e}")
                print("继续监控...")
            
            # 等待下次刷新
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
        print("=" * 80)

if __name__ == "__main__":
    main()

