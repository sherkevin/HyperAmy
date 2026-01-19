#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HyperAmy并行索引 - 实时进度监控
支持动态刷新显示索引进度、速度、预计完成时间等
"""
import subprocess
import time
import re
import os
from datetime import datetime

def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_process_info(server="hyperamy-server"):
    """获取进程信息"""
    cmd = f'ssh {server} "ps aux | grep \'[p]ython.*test_hyperamy_parallel.py\' | grep python"'
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
    cmd = f'ssh {server} "cd /public/jiangh/HyperAmy && tail -50 test_hyperamy_parallel.log 2>/dev/null"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                # 匹配进度条格式: 84%|████████▍ | 8400/10000 [32:54<06:32,  4.08it/s]
                if '提取情绪向量（并发+GPU）:' in line and '%|' in line:
                    match = re.search(r'(\d+)%\|.*?(\d+)/(\d+).*?\[(.*?)<(.*?), (.*?)\]', line)
                    if match:
                        percent = int(match.group(1))
                        current = int(match.group(2))
                        total = int(match.group(3))
                        elapsed = match.group(4)
                        remaining = match.group(5)
                        speed = match.group(6)
                        return {
                            'percent': percent,
                            'current': current,
                            'total': total,
                            'elapsed': elapsed,
                            'remaining': remaining,
                            'speed': speed,
                            'task_name': '提取情绪向量（并发+GPU）',
                            'found': True
                        }
            # 检查是否已完成
            for line in reversed(lines):
                if '✅' in line and ('完成' in line or '存储了' in line):
                    return {'found': True, 'completed': True, 'message': line.strip()}
    except:
        pass
    return {'found': False}

def format_progress_bar(percent, width=50):
    """格式化进度条"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '▌' * (width - filled)
    return bar

def format_time(time_str):
    """格式化时间字符串"""
    return time_str

def main():
    server = "hyperamy-server"
    refresh_interval = 3
    
    print("🚀 启动 HyperAmy 并行索引实时监控...")
    print(f"📡 连接服务器: {server}")
    print(f"🔄 刷新间隔: {refresh_interval}秒")
    print("💡 提示: 按 Ctrl+C 退出监控\n")
    
    refresh_count = 0
    
    try:
        while True:
            refresh_count += 1
            clear_screen()
            
            # 获取进程信息
            process_info = get_process_info(server)
            
            # 获取进度信息
            progress_info = get_latest_progress(server)
            
            # 显示标题
            print("=" * 80)
            print("📊 HyperAmy 并行索引 - 实时进度监控")
            print("=" * 80)
            print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 刷新次数: {refresh_count}\n")
            
            # 显示进程状态
            print("🔄 进程状态:")
            print("-" * 80)
            if process_info.get('running'):
                print(f"  ✅ 进程运行中")
                print(f"     PID: {process_info.get('pid')}")
                print(f"     CPU: {process_info.get('cpu')}%")
                print(f"     内存: {process_info.get('mem')}%")
                print(f"     运行时间: {process_info.get('time')}")
            else:
                print("  ⏸️  进程未运行")
            print()
            
            # 显示进度
            print("📈 当前进度:")
            print("-" * 80)
            if progress_info.get('found'):
                if progress_info.get('completed'):
                    print(f"  ✅ {progress_info.get('message', '索引完成')}")
                else:
                    percent = progress_info.get('percent', 0)
                    current = progress_info.get('current', 0)
                    total = progress_info.get('total', 0)
                    elapsed = progress_info.get('elapsed', 'N/A')
                    remaining = progress_info.get('remaining', 'N/A')
                    speed = progress_info.get('speed', 'N/A')
                    task_name = progress_info.get('task_name', '未知任务')
                    
                    print(f"  任务: {task_name}")
                    print(f"  进度: {percent}% ({current}/{total})")
                    bar = format_progress_bar(percent)
                    print(f"  [{bar}] {percent}%")
                    print(f"  已用时间: {elapsed}")
                    print(f"  预计剩余: {remaining}")
                    print(f"  处理速度: {speed}")
                    
                    # 计算预计完成时间
                    try:
                        remaining_parts = remaining.split(':')
                        if len(remaining_parts) == 2:
                            if int(remaining_parts[0]) < 10:  # MM:SS格式
                                remaining_minutes = int(remaining_parts[0]) + int(remaining_parts[1]) / 60
                            else:  # HH:MM格式
                                remaining_minutes = int(remaining_parts[0]) * 60 + int(remaining_parts[1])
                            print(f"  预计还需: 约 {remaining_minutes:.1f} 分钟")
                    except:
                        pass
            else:
                print("  ⏳ 等待进度信息...")
            print()
            
            # 显示提示
            print("=" * 80)
            print("💡 提示: 按 Ctrl+C 退出监控")
            print("=" * 80)
            
            # 如果已完成，退出循环
            if progress_info.get('completed'):
                print("\n🎉 HyperAmy 并行索引已完成！")
                break
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已退出")
    except Exception as e:
        print(f"\n❌ 监控出错: {e}")

if __name__ == '__main__':
    main()
