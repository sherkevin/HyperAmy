#!/usr/bin/env python3
"""
实验实时监控系统
持续监控实验进度、日志、资源使用情况
"""
import os
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime
import sys

def get_latest_log_file(log_dir="logs"):
    """获取最新的实验日志文件"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
    
    log_files = sorted(log_path.glob("got_experiment_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    return log_files[0] if log_files else None

def get_process_status(pid_file="got_experiment.pid"):
    """检查实验进程状态"""
    pid_path = Path(pid_file)
    if not pid_path.exists():
        return None, None
    
    try:
        pid = int(pid_path.read_text().strip())
        # 检查进程是否存在
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime="],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            etime = result.stdout.strip()
            return pid, etime
        else:
            return None, None
    except Exception:
        return None, None

def parse_progress_from_log(log_file):
    """从日志文件解析进度"""
    if not log_file or not log_file.exists():
        return {}
    
    try:
        content = log_file.read_text(encoding='utf-8', errors='ignore')
        
        progress = {
            'hipporag_index': False,
            'hipporag_retrieve': False,
            'fusion_index': False,
            'fusion_retrieve': False,
            'hyperamy_retrieve': False,
            'completed': False,
            'hipporag_metrics': None,
            'fusion_metrics': None,
            'hyperamy_metrics': None,
            'forgotten_count': 0,
            'total_scoring': 0,
            'zero_results': 0,
            'hyperamy_progress': None,
        }
        
        # 检查各步骤完成状态
        if re.search(r'✅.*HippoRAG.*索引.*完成|HippoRAG.*索引.*完成', content):
            progress['hipporag_index'] = True
        if re.search(r'✅.*HippoRAG.*检索.*完成|HippoRAG.*检索.*完成', content):
            progress['hipporag_retrieve'] = True
            # 提取HippoRAG指标
            match = re.search(r"HippoRAG.*检索完成.*?\n.*?Recall@1['\"]?\s*[:=]\s*([\d.]+)", content)
            if match:
                progress['hipporag_metrics'] = float(match.group(1))
        if re.search(r'✅.*Fusion.*索引.*完成|Fusion.*索引.*完成', content):
            progress['fusion_index'] = True
        if re.search(r'✅.*Fusion.*检索.*完成|Fusion.*检索.*完成', content):
            progress['fusion_retrieve'] = True
            # 提取Fusion指标
            match = re.search(r"Fusion.*检索完成.*?\n.*?Recall@1['\"]?\s*[:=]\s*([\d.]+)", content)
            if match:
                progress['fusion_metrics'] = float(match.group(1))
        if re.search(r'✅.*HyperAmy.*检索.*完成|HyperAmy.*检索.*完成|HyperAmy 评估指标', content):
            progress['hyperamy_retrieve'] = True
            # 提取HyperAmy指标
            match = re.search(r"HyperAmy.*评估指标.*?\n.*?Recall@1['\"]?\s*[:=]\s*([\d.]+)", content)
            if match:
                progress['hyperamy_metrics'] = float(match.group(1))
        if re.search(r'✅.*实验完成|实验全部完成|所有检索完成', content):
            progress['completed'] = True
        
        # 统计遗忘问题
        progress['forgotten_count'] = len(re.findall(r'forgotten -> 0 results', content))
        
        # 统计Thermodynamic scoring
        progress['total_scoring'] = len(re.findall(r'Thermodynamic scoring', content))
        progress['zero_results'] = progress['forgotten_count']
        
        # HyperAmy检索进度
        hyperamy_matches = re.findall(r'HyperAmy检索:\s*(\d+)%', content)
        if hyperamy_matches:
            progress['hyperamy_progress'] = int(hyperamy_matches[-1])
        
        return progress
    
    except Exception as e:
        return {'error': str(e)}

def get_system_resources():
    """获取系统资源使用情况"""
    try:
        # 内存使用
        result = subprocess.run(
            ["free", "-h"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                mem_line = lines[1].split()
                if len(mem_line) >= 7:
                    return {
                        'memory_used': mem_line[2],
                        'memory_total': mem_line[1],
                        'memory_available': mem_line[6]
                    }
        return {'memory_used': 'N/A', 'memory_total': 'N/A', 'memory_available': 'N/A'}
    except Exception:
        return {'memory_used': 'N/A', 'memory_total': 'N/A', 'memory_available': 'N/A'}

def get_log_tail(log_file, n=10):
    """获取日志最后n行"""
    if not log_file or not log_file.exists():
        return []
    
    try:
        result = subprocess.run(
            ["tail", "-n", str(n), str(log_file)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except Exception:
        return []

def display_progress(pid, etime, progress, resources, log_file):
    """显示进度信息"""
    # 清屏
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 80)
    print(f"实验实时监控系统 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # 进程状态
    if pid:
        print(f"✅ 实验进程: {pid} | 运行时长: {etime}")
    else:
        print("❌ 实验进程未运行")
    
    if log_file:
        log_size = log_file.stat().st_size / 1024  # KB
        print(f"📄 日志文件: {log_file.name} ({log_size:.1f} KB)")
    print()
    
    # 步骤进度
    print("─" * 80)
    print("📊 实验进度:")
    print("─" * 80)
    
    steps = [
        ("[1] HippoRAG索引", progress.get('hipporag_index', False)),
        ("[2] HippoRAG检索", progress.get('hipporag_retrieve', False)),
        ("[3] Fusion索引", progress.get('fusion_index', False)),
        ("[4] Fusion检索", progress.get('fusion_retrieve', False)),
        ("[5] HyperAmy检索", progress.get('hyperamy_retrieve', False)),
    ]
    
    for step_name, completed in steps:
        status = "✅" if completed else "⏳"
        print(f"  {status} {step_name}")
        
        # 显示指标
        if step_name == "[2] HippoRAG检索" and progress.get('hipporag_metrics') is not None:
            print(f"      Recall@1: {progress['hipporag_metrics']:.4f}")
        if step_name == "[4] Fusion检索" and progress.get('fusion_metrics') is not None:
            print(f"      Recall@1: {progress['fusion_metrics']:.4f}")
        if step_name == "[5] HyperAmy检索":
            if progress.get('hyperamy_metrics') is not None:
                print(f"      Recall@1: {progress['hyperamy_metrics']:.4f}")
            elif progress.get('hyperamy_progress') is not None:
                print(f"      进度: {progress['hyperamy_progress']}%")
    
    if progress.get('completed', False):
        print()
        print("🎉🎉🎉 实验全部完成！🎉🎉🎉")
    
    print()
    
    # HyperAmy修复验证
    print("─" * 80)
    print("🔍 HyperAmy修复验证:")
    print("─" * 80)
    
    forgotten_count = progress.get('forgotten_count', 0)
    total_scoring = progress.get('total_scoring', 0)
    zero_results = progress.get('zero_results', 0)
    non_zero_results = total_scoring - zero_results
    
    if total_scoring > 0:
        print(f"  Thermodynamic Scoring次数: {total_scoring}")
        print(f"  0结果次数: {zero_results}")
        print(f"  有结果次数: {non_zero_results}")
        
        if forgotten_count == 0 and total_scoring > 0:
            print("  ✅ 修复生效：未发现遗忘问题")
        elif forgotten_count < total_scoring:
            print(f"  ⚠️  部分生效：{non_zero_results}/{total_scoring} 查询返回结果")
        else:
            print(f"  ❌ 仍有问题：所有查询返回0结果")
    else:
        if progress.get('hyperamy_retrieve', False):
            print("  ⏳ HyperAmy检索已完成，等待最终统计")
        else:
            print("  ⏳ 尚未到达HyperAmy检索阶段")
    
    print()
    
    # 系统资源
    print("─" * 80)
    print("💾 系统资源:")
    print("─" * 80)
    print(f"  内存: {resources.get('memory_used', 'N/A')} / {resources.get('memory_total', 'N/A')} "
          f"(可用: {resources.get('memory_available', 'N/A')})")
    print()
    
    # 最新日志
    print("─" * 80)
    print("📋 最新日志 (最后8行):")
    print("─" * 80)
    log_lines = get_log_tail(log_file, 8)
    for line in log_lines:
        # 截断过长的行
        if len(line) > 100:
            line = line[:97] + "..."
        print(f"  {line}")
    
    print()
    print("=" * 80)
    print("按 Ctrl+C 退出监控")

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='实验实时监控系统')
    parser.add_argument('--interval', type=int, default=5, help='刷新间隔（秒）')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--pid-file', type=str, default='got_experiment.pid', help='PID文件路径')
    args = parser.parse_args()
    
    print("启动实验实时监控系统...")
    print(f"刷新间隔: {args.interval}秒")
    print("按 Ctrl+C 退出")
    time.sleep(2)
    
    try:
        while True:
            # 获取状态
            pid, etime = get_process_status(args.pid_file)
            log_file = get_latest_log_file(args.log_dir)
            progress = parse_progress_from_log(log_file)
            resources = get_system_resources()
            
            # 显示进度
            display_progress(pid, etime, progress, resources, log_file)
            
            # 如果实验完成，降低刷新频率
            if progress.get('completed', False):
                time.sleep(args.interval * 2)
            else:
                time.sleep(args.interval)
                
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        sys.exit(0)

if __name__ == '__main__':
    main()
