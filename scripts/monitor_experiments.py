#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时监控两批实验的进度

使用方法:
    python scripts/monitor_experiments.py
"""
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

def get_v1_progress(log_file: Path) -> Tuple[int, int, float, Optional[str]]:
    """获取第一批实验的进度"""
    if not log_file.exists():
        return 0, 0, 0.0, None
    
    try:
        # 读取日志文件的最后几行
        result = subprocess.run(
            ['tail', '-50', str(log_file)],
            capture_output=True,
            text=True,
            timeout=5
        )
        lines = result.stdout.split('\n')
        
        # 查找进度条行
        progress_line = None
        for line in reversed(lines):
            if 'Extracting emotion vectors:' in line:
                progress_line = line
                break
        
        if not progress_line:
            # 检查是否已完成
            for line in reversed(lines):
                if '实验完成' in line or '实验失败' in line or 'Error' in line:
                    return 468, 468, 100.0, line.strip()
                if '索引完成' in line or '检索完成' in line:
                    return 468, 468, 100.0, "已完成"
            return 0, 0, 0.0, None
        
        # 解析进度：例如 "Extracting emotion vectors:  92%|█████████▏| 430/468 [30:30<02:28,  3.91s/it]"
        import re
        match = re.search(r'(\d+)%', progress_line)
        if match:
            percent = int(match.group(1))
            current = int(percent * 468 / 100)
            return current, 468, percent, progress_line.strip()
        
        # 尝试另一种格式：例如 "430/468"
        match = re.search(r'(\d+)/(\d+)', progress_line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            percent = (current / total) * 100 if total > 0 else 0
            return current, total, percent, progress_line.strip()
        
        return 0, 0, 0.0, progress_line.strip()
    
    except Exception as e:
        return 0, 0, 0.0, f"Error: {str(e)}"

def get_v2_status(result_file: Path, log_file: Path) -> Tuple[str, Optional[dict]]:
    """获取第二批实验的状态"""
    if result_file.exists():
        try:
            import json
            with open(result_file) as f:
                results = json.load(f)
            
            hipporag_ok = any(r.get('hipporag', {}).get('available', False) for r in results)
            fusion_ok = any(r.get('fusion', {}).get('available', False) for r in results)
            
            status = "✅ 已完成"
            if hipporag_ok and fusion_ok:
                status += " (HippoRAG + Fusion 均成功)"
            elif hipporag_ok or fusion_ok:
                status += " (部分成功)"
            
            return status, {'results_count': len(results), 'hipporag_ok': hipporag_ok, 'fusion_ok': fusion_ok}
        except:
            return "✅ 已完成 (结果文件存在)", None
    elif log_file.exists():
        # 检查日志最后几行
        try:
            result = subprocess.run(
                ['tail', '-20', str(log_file)],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = result.stdout
            
            if '实验完成' in lines or '已完成' in lines:
                return "✅ 已完成", None
            elif '错误' in lines or 'Error' in lines or '失败' in lines:
                return "❌ 失败", None
            else:
                return "⏳ 进行中...", None
        except:
            return "⏳ 状态未知", None
    else:
        return "⏳ 未开始", None

def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)}分{int(seconds % 60)}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分钟"

def draw_progress_bar(current: int, total: int, width: int = 50) -> str:
    """绘制进度条"""
    if total == 0:
        return "[" + " " * width + "]"
    
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total} ({percent*100:.1f}%)"

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='实时监控两批实验的进度')
    parser.add_argument('--interval', type=float, default=2.0, help='更新间隔（秒）')
    parser.add_argument('--server', type=str, default='hyperamy-server', help='服务器地址')
    args = parser.parse_args()
    
    # 日志和结果文件路径
    v1_log = PROJECT_ROOT / 'test_two_methods_comparison.log'
    v2_log = PROJECT_ROOT / 'test_two_methods_comparison_v2.log'
    v2_result = PROJECT_ROOT / 'outputs' / 'two_methods_comparison_v2' / 'comparison_results.json'
    
    print("=" * 80)
    print("🔍 实验实时监控（按 Ctrl+C 退出）")
    print("=" * 80)
    print(f"监控间隔: {args.interval}秒")
    print(f"服务器: {args.server}")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    try:
        while True:
            # 清屏（在终端中）
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("=" * 80)
            print(f"🔍 实验实时监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print(f"运行时长: {format_time(time.time() - start_time)}")
            print()
            
            # 第一批实验状态
            print("📊 第一批实验 (V1 - 原始版本)")
            print("-" * 80)
            
            # 如果是远程服务器，需要通过SSH获取
            if args.server:
                try:
                    # 获取进度
                    cmd = f"ssh {args.server} 'cd /public/jiangh/HyperAmy && tail -50 test_two_methods_comparison.log 2>/dev/null | grep \"Extracting emotion vectors\" | tail -1'"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                    progress_line = result.stdout.strip()
                    
                    if progress_line:
                        import re
                        match = re.search(r'(\d+)%', progress_line)
                        if match:
                            percent = int(match.group(1))
                            current = int(percent * 468 / 100)
                            total = 468
                            
                            print(f"状态: 🟢 正在运行")
                            print(f"进度: {draw_progress_bar(current, total)}")
                            print(f"当前: {current}/{total} ({percent}%)")
                            
                            # 估算剩余时间
                            if 'it/s' in progress_line:
                                match_speed = re.search(r'(\d+\.?\d*)s/it', progress_line)
                                if match_speed:
                                    speed = float(match_speed.group(1))
                                    remaining = speed * (total - current)
                                    print(f"速度: {speed:.2f}秒/文档")
                                    print(f"预计剩余时间: {format_time(remaining)}")
                        else:
                            print(f"进度信息: {progress_line[:80]}")
                    else:
                        # 检查是否完成
                        cmd = f"ssh {args.server} 'cd /public/jiangh/HyperAmy && tail -10 test_two_methods_comparison.log 2>/dev/null | tail -3'"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                        last_lines = result.stdout.strip()
                        
                        if '实验完成' in last_lines or '已完成' in last_lines:
                            print("状态: ✅ 已完成")
                        elif last_lines:
                            print(f"状态: 🟢 运行中")
                            print(f"最新日志: {last_lines.split(chr(10))[-1][:80]}")
                        else:
                            print("状态: ⏳ 等待中...")
                    
                    # 检查进程
                    cmd = f"ssh {args.server} 'ps aux | grep \"[t]est_two_methods_comparison.py\" | grep -v v2 | head -1'"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                    if not result.stdout.strip():
                        print("⚠️  进程未运行")
                    
                except Exception as e:
                    print(f"❌ 获取V1状态失败: {e}")
            else:
                # 本地监控
                current, total, percent, progress_line = get_v1_progress(v1_log)
                if current > 0:
                    print(f"状态: 🟢 正在运行")
                    print(f"进度: {draw_progress_bar(current, total)}")
                    if progress_line:
                        print(f"详情: {progress_line[:80]}")
                elif v1_log.exists():
                    print("状态: ✅ 已完成或等待中")
                else:
                    print("状态: ⏳ 未开始")
            
            print()
            
            # 第二批实验状态
            print("📊 第二批实验 (V2 - 优化版本)")
            print("-" * 80)
            
            if args.server:
                try:
                    # 检查结果文件
                    cmd = f"ssh {args.server} 'test -f /public/jiangh/HyperAmy/outputs/two_methods_comparison_v2/comparison_results.json && echo \"exists\" || echo \"not_exists\"'"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                    
                    if 'exists' in result.stdout:
                        # 获取结果统计
                        python_script = """
import json
with open('outputs/two_methods_comparison_v2/comparison_results.json') as f:
    results = json.load(f)
print(f'结果数: {len(results)}')
hipporag_ok = sum(1 for r in results if r.get('hipporag', {}).get('available', False))
fusion_ok = sum(1 for r in results if r.get('fusion', {}).get('available', False))
print(f'HippoRAG成功: {hipporag_ok}/{len(results)}')
print(f'Fusion成功: {fusion_ok}/{len(results)}')
"""
                        cmd = f"ssh {args.server} 'cd /public/jiangh/HyperAmy && python3'"
                        result = subprocess.run(
                            cmd, 
                            input=python_script,
                            shell=True, 
                            capture_output=True, 
                            text=True, 
                            timeout=5
                        )
                        
                        print("状态: ✅ 已完成")
                        if result.stdout:
                            for line in result.stdout.strip().split('\n'):
                                if line.strip():
                                    print(f"  {line.strip()}")
                    else:
                        # 检查日志
                        cmd = f"ssh {args.server} 'cd /public/jiangh/HyperAmy && tail -5 test_two_methods_comparison_v2.log 2>/dev/null | tail -3'"
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                        last_lines = result.stdout.strip()
                        
                        if last_lines:
                            print("状态: ⏳ 运行中")
                            print(f"最新日志: {last_lines.split(chr(10))[-1][:80]}")
                        else:
                            print("状态: ⏳ 未开始或状态未知")
                
                except Exception as e:
                    print(f"❌ 获取V2状态失败: {e}")
            else:
                # 本地监控
                status, info = get_v2_status(v2_result, v2_log)
                print(f"状态: {status}")
                if info:
                    print(f"结果数: {info.get('results_count', 0)}")
                    print(f"HippoRAG: {'✅' if info.get('hipporag_ok') else '❌'}")
                    print(f"Fusion: {'✅' if info.get('fusion_ok') else '❌'}")
            
            print()
            print("=" * 80)
            print(f"下次更新: {args.interval}秒后（按 Ctrl+C 退出）")
            
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("👋 监控已停止")
        print("=" * 80)

if __name__ == '__main__':
    main()

