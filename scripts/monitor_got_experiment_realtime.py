#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GoT实验实时监控系统
实时显示实验进度、资源使用、错误信息等
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

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
    WHITE = '\033[97m'

def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_process_info(pid_file: str) -> Optional[Dict]:
    """获取进程信息"""
    try:
        if not os.path.exists(pid_file):
            return None
        
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # 检查进程是否运行
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'etime=,pid=,%cpu=,%mem='],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode != 0:
            return {'running': False, 'pid': pid}
        
        parts = result.stdout.strip().split()
        if len(parts) >= 4:
            etime = parts[0]
            cpu = parts[2]
            mem = parts[3]
            return {
                'running': True,
                'pid': pid,
                'etime': etime,
                'cpu': cpu,
                'mem': mem
            }
        
        return {'running': True, 'pid': pid}
    except Exception as e:
        return None

def get_latest_log_file(log_dir: str = "logs") -> Optional[str]:
    """获取最新的日志文件"""
    try:
        log_path = Path(log_dir)
        if not log_path.exists():
            return None
        
        log_files = list(log_path.glob("got_experiment_*.log"))
        if not log_files:
            return None
        
        # 按修改时间排序，返回最新的
        latest = max(log_files, key=lambda p: p.stat().st_mtime)
        return str(latest)
    except Exception:
        return None

def parse_log_progress(log_file: str) -> Dict:
    """解析日志获取进度信息"""
    if not log_file or not os.path.exists(log_file):
        return {
            'current_stage': '未开始',
            'current_step': 0,
            'total_steps': 0,
            'progress_percent': 0,
            'last_update': None
        }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 获取最后500行进行分析
        recent_lines = lines[-500:] if len(lines) > 500 else lines
        log_text = ''.join(recent_lines)
        
        # 步骤完成状态
        steps_status = {
            'hipporag_index': False,
            'hipporag_retrieve': False,
            'fusion_index': False,
            'fusion_retrieve': False,
            'hyperamy_retrieve': False
        }
        
        # 检查各步骤完成状态
        if 'HippoRAG.*索引.*完成' in log_text or 'HippoRAG 索引完成' in log_text:
            steps_status['hipporag_index'] = True
        if 'HippoRAG.*检索.*完成' in log_text or 'HippoRAG 检索完成' in log_text:
            steps_status['hipporag_retrieve'] = True
        if 'Fusion.*索引.*完成' in log_text or 'Fusion 索引完成' in log_text:
            steps_status['fusion_index'] = True
        if 'Fusion.*检索.*完成' in log_text or 'Fusion 检索完成' in log_text:
            steps_status['fusion_retrieve'] = True
        if 'HyperAmy.*检索.*完成' in log_text or 'HyperAmy 评估指标' in log_text:
            steps_status['hyperamy_retrieve'] = True
        
        # 当前阶段判断
        current_stage = '等待开始'
        current_step = 0
        total_steps = 5
        
        if steps_status['hyperamy_retrieve']:
            current_stage = '✅ 全部完成'
            current_step = 5
        elif steps_status['fusion_retrieve']:
            current_stage = '🔄 HyperAmy检索'
            current_step = 4
            # 查找HyperAmy检索进度
            for line in reversed(recent_lines):
                if 'HyperAmy检索:' in line:
                    # 提取进度百分比，例如 "38%|███▊      | 19/50"
                    import re
                    match = re.search(r'(\d+)%', line)
                    if match:
                        current_stage = f'🔄 HyperAmy检索: {match.group(1)}%'
                    break
        elif steps_status['fusion_index']:
            current_stage = '🔄 Fusion检索'
            current_step = 3
        elif steps_status['hipporag_retrieve']:
            current_stage = '🔄 Fusion索引'
            current_step = 2
        elif steps_status['hipporag_index']:
            current_stage = '🔄 HippoRAG检索'
            current_step = 1
        else:
            current_stage = '🔄 HippoRAG索引'
            current_step = 0
        
        # 获取文件修改时间
        last_update = datetime.fromtimestamp(os.path.getmtime(log_file)).strftime('%H:%M:%S')
        
        return {
            'current_stage': current_stage,
            'current_step': current_step,
            'total_steps': total_steps,
            'progress_percent': (current_step / total_steps) * 100 if total_steps > 0 else 0,
            'last_update': last_update
        }
    except Exception as e:
        return {
            'current_stage': f'解析错误: {str(e)[:30]}',
            'current_step': 0,
            'total_steps': 0,
            'progress_percent': 0,
            'last_update': None
        }

def get_error_stats(log_file: str) -> Dict:
    """获取错误统计"""
    if not log_file or not os.path.exists(log_file):
        return {'error_count': 0, 'warning_count': 0, 'latest_errors': []}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        error_lines = [line for line in lines if any(kw in line.upper() for kw in ['ERROR', 'EXCEPTION', 'TRACEBACK', '失败'])]
        warning_lines = [line for line in lines if 'WARNING' in line.upper() and 'ERROR' not in line.upper()]
        
        return {
            'error_count': len(error_lines),
            'warning_count': len(warning_lines),
            'latest_errors': error_lines[-3:] if error_lines else [],
            'latest_warnings': warning_lines[-3:] if warning_lines else []
        }
    except Exception:
        return {'error_count': 0, 'warning_count': 0, 'latest_errors': []}

def get_forgotten_stats(log_file: str) -> Dict:
    """获取遗忘统计（HyperAmy修复验证）"""
    if not log_file or not os.path.exists(log_file):
        return {'total': 0, 'zero_results': 0, 'has_results': 0}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 统计Thermodynamic scoring
        import re
        scoring_matches = re.findall(r'Thermodynamic scoring.*?(\d+) input.*?(\d+) forgotten.*?(\d+) results', content)
        
        total = len(scoring_matches)
        zero_results = sum(1 for m in scoring_matches if int(m[2]) == 0)
        has_results = total - zero_results
        
        return {
            'total': total,
            'zero_results': zero_results,
            'has_results': has_results
        }
    except Exception:
        return {'total': 0, 'zero_results': 0, 'has_results': 0}

def get_recent_logs(log_file: str, n: int = 10) -> List[str]:
    """获取最近的日志行"""
    if not log_file or not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [line.rstrip() for line in lines[-n:]]
    except Exception:
        return []

def draw_progress_bar(current: int, total: int, width: int = 50) -> str:
    """绘制进度条"""
    if total == 0:
        return '[░░░░░░░░░░░░░░░░░░░░] 0.0% (0/0)'
    
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}] {percent*100:.1f}% ({current}/{total})'

def get_system_resources() -> Dict:
    """获取系统资源使用"""
    try:
        result = subprocess.run(
            ['free', '-h'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                mem_line = lines[1].split()
                if len(mem_line) >= 7:
                    return {
                        'total': mem_line[1],
                        'used': mem_line[2],
                        'available': mem_line[6]
                    }
    except Exception:
        pass
    
    return {'total': 'N/A', 'used': 'N/A', 'available': 'N/A'}

def display_dashboard(pid_file: str, log_file: str, update_count: int = 0):
    """显示监控仪表板"""
    clear_screen()
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}GoT实验实时监控系统{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (更新 #{update_count})")
    print()
    
    # 1. 进程状态
    print(f"{Colors.BOLD}【1】进程状态{Colors.RESET}")
    proc_info = get_process_info(pid_file)
    if proc_info and proc_info.get('running'):
        print(f"  {Colors.GREEN}✅ 运行中{Colors.RESET} (PID: {proc_info['pid']})")
        print(f"  运行时间: {proc_info.get('etime', 'N/A')}")
        print(f"  CPU使用: {Colors.YELLOW}{proc_info.get('cpu', 'N/A')}%{Colors.RESET}")
        print(f"  内存使用: {Colors.YELLOW}{proc_info.get('mem', 'N/A')}%{Colors.RESET}")
    else:
        print(f"  {Colors.RED}❌ 进程未运行{Colors.RESET}")
        if proc_info:
            print(f"  PID文件存在但进程已停止 (PID: {proc_info.get('pid', 'N/A')})")
    print()
    
    # 2. 实验进度
    print(f"{Colors.BOLD}【2】实验进度{Colors.RESET}")
    if log_file:
        progress = parse_log_progress(log_file)
        print(f"  当前阶段: {Colors.YELLOW}{progress['current_stage']}{Colors.RESET}")
        if progress['total_steps'] > 0:
            print(f"  进度: {draw_progress_bar(progress['current_step'], progress['total_steps'])}")
        if progress['last_update']:
            print(f"  日志更新: {progress['last_update']}")
        print(f"  日志文件: {os.path.basename(log_file)}")
    else:
        print(f"  {Colors.RED}未找到日志文件{Colors.RESET}")
    print()
    
    # 3. HyperAmy修复验证
    print(f"{Colors.BOLD}【3】HyperAmy修复验证{Colors.RESET}")
    if log_file:
        forgotten_stats = get_forgotten_stats(log_file)
        total = forgotten_stats['total']
        zero_results = forgotten_stats['zero_results']
        has_results = forgotten_stats['has_results']
        
        if total > 0:
            if zero_results == 0:
                print(f"  {Colors.GREEN}✅ 完美！所有查询都有结果{Colors.RESET}")
                print(f"  总计: {total} 次评分，全部有结果")
            elif has_results > 0:
                print(f"  {Colors.YELLOW}⚠️  部分查询有结果{Colors.RESET}")
                print(f"  总计: {total} 次评分，{has_results} 次有结果，{zero_results} 次0结果")
            else:
                print(f"  {Colors.RED}❌ 所有查询仍返回0结果{Colors.RESET}")
                print(f"  总计: {total} 次评分，全部0结果（需要进一步检查）")
        else:
            print(f"  {Colors.CYAN}⏳ 尚未到达Thermodynamic scoring阶段{Colors.RESET}")
    else:
        print(f"  {Colors.RED}未找到日志文件{Colors.RESET}")
    print()
    
    # 4. 错误统计
    print(f"{Colors.BOLD}【4】错误统计{Colors.RESET}")
    if log_file:
        error_stats = get_error_stats(log_file)
        error_count = error_stats['error_count']
        warning_count = error_stats['warning_count']
        
        if error_count > 0:
            print(f"  {Colors.RED}⚠️  错误: {error_count} 个{Colors.RESET}")
            if error_stats['latest_errors']:
                print(f"  最新错误:")
                for err in error_stats['latest_errors'][-2:]:
                    print(f"    {Colors.RED}{err[:100]}{Colors.RESET}")
        else:
            print(f"  {Colors.GREEN}✅ 无错误{Colors.RESET}")
        
        if warning_count > 0:
            print(f"  {Colors.YELLOW}⚠️  警告: {warning_count} 个{Colors.RESET}")
    else:
        print(f"  {Colors.RED}未找到日志文件{Colors.RESET}")
    print()
    
    # 5. 系统资源
    print(f"{Colors.BOLD}【5】系统资源{Colors.RESET}")
    resources = get_system_resources()
    print(f"  内存: {resources['used']} / {resources['total']} (可用: {resources['available']})")
    print()
    
    # 6. 最新日志
    print(f"{Colors.BOLD}【6】最新日志（最后8行）{Colors.RESET}")
    if log_file:
        recent_logs = get_recent_logs(log_file, 8)
        for log_line in recent_logs:
            # 高亮错误和警告
            if any(kw in log_line.upper() for kw in ['ERROR', 'EXCEPTION', '失败']):
                print(f"  {Colors.RED}{log_line[:120]}{Colors.RESET}")
            elif 'WARNING' in log_line.upper():
                print(f"  {Colors.YELLOW}{log_line[:120]}{Colors.RESET}")
            elif 'forgotten -> 0 results' in log_line:
                print(f"  {Colors.RED}{log_line[:120]}{Colors.RESET}")
            else:
                print(f"  {log_line[:120]}")
    else:
        print(f"  {Colors.RED}未找到日志文件{Colors.RESET}")
    
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}💡 提示: 按 Ctrl+C 退出监控{Colors.RESET}")
    print(f"{Colors.CYAN}⏱️  自动刷新间隔: 3秒{Colors.RESET}")

def main():
    """主函数"""
    # 默认配置
    pid_file = "got_experiment.pid"
    log_dir = "logs"
    refresh_interval = 3  # 3秒刷新一次
    
    # 如果通过SSH连接，使用远程路径
    if 'SSH_CLIENT' in os.environ or 'SSH_CONNECTION' in os.environ:
        # 在服务器上运行
        base_dir = "/public/jiangh/HyperAmy"
        pid_file = os.path.join(base_dir, pid_file)
        log_dir = os.path.join(base_dir, log_dir)
    
    update_count = 0
    
    print(f"{Colors.BOLD}{Colors.CYAN}🚀 GoT实验实时监控系统启动{Colors.RESET}")
    print(f"PID文件: {pid_file}")
    print(f"日志目录: {log_dir}")
    print(f"刷新间隔: {refresh_interval}秒")
    print(f"{Colors.CYAN}💡 提示: 按 Ctrl+C 退出监控{Colors.RESET}")
    print()
    time.sleep(2)
    
    try:
        while True:
            update_count += 1
            log_file = get_latest_log_file(log_dir)
            display_dashboard(pid_file, log_file, update_count)
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}监控已停止{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}监控出错: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
