#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器资源监控和预警系统
- 监控CPU、内存、GPU使用率
- 当资源使用过高时自动预警
- 自动检测并清理异常进程
- 确保训练任务使用GPU而不是CPU
"""

import subprocess
import time
import json
import logging
import smtplib
import signal
import sys
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resource_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 预警阈值
THRESHOLDS = {
    'cpu_percent': 90.0,      # CPU使用率超过90%预警
    'memory_percent': 90.0,   # 内存使用率超过90%预警
    'cpu_warn_duration': 60,  # CPU持续高负载1分钟预警（秒）
    'memory_warn_duration': 60,  # 内存持续高负载1分钟预警（秒）
    'auto_stop': True,        # 是否自动停止任务
    'auto_stop_duration': 120,  # 超过阈值后持续多久自动停止（秒）
}

# 进程清理阈值
CLEANUP_THRESHOLDS = {
    'cpu_percent': 90.0,      # CPU使用率超过90%自动停止
    'memory_percent': 90.0,   # 内存使用率超过90%自动停止
    'check_interval': 30,     # 检查间隔（秒）- 更频繁检查
}

# 应该使用GPU的训练进程关键词
GPU_TRAINING_KEYWORDS = ['train.py', 'train_emos', 'python.*train']
CPU_ONLY_PATTERNS = ['jupyter', 'notebook']  # 允许CPU运行的进程


class ResourceMonitor:
    def __init__(self, server_host: str = '10.103.92.120', 
                 server_port: int = 1066, 
                 server_user: str = 'jiangh'):
        self.server_host = server_host
        self.server_port = server_port
        self.server_user = server_user
        self.high_cpu_start = None
        self.high_memory_start = None
        self.alert_sent = False
        self.stop_triggered = False
        self.monitored_pids = []  # 需要监控的训练进程PID列表
        
    def execute_remote(self, command: str) -> tuple[str, int]:
        """执行远程命令"""
        ssh_cmd = [
            'ssh', '-p', str(self.server_port),
            '-o', 'ConnectTimeout=10',
            '-o', 'StrictHostKeyChecking=no',
            f'{self.server_user}@{self.server_host}',
            command
        ]
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip(), result.returncode
        except subprocess.TimeoutExpired:
            logger.error(f"SSH命令超时: {command}")
            return "", 1
        except Exception as e:
            logger.error(f"执行SSH命令失败: {e}")
            return "", 1
    
    def get_cpu_memory_usage(self) -> Dict:
        """获取CPU和内存使用率"""
        # 使用top命令获取实时CPU和内存使用率
        cmd = "top -bn1 | head -5 | tail -2"
        output, code = self.execute_remote(cmd)
        
        if code != 0:
            # 备用方案：使用/proc/meminfo
            cmd = "cat /proc/meminfo | head -3 && echo '---' && grep 'cpu ' /proc/stat | awk '{usage=100-($5*100/($2+$3+$4+$5+$6+$7+$8))} END {print usage}'"
            output, code = self.execute_remote(cmd)
        
        # 解析内存使用率
        cmd = "free | grep Mem | awk '{printf \"%.1f\", $3/$2 * 100.0}'"
        memory_percent, _ = self.execute_remote(cmd)
        
        # 解析CPU使用率
        cmd = "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'"
        cpu_percent, _ = self.execute_remote(cmd)
        
        try:
            cpu = float(cpu_percent) if cpu_percent else 0.0
            memory = float(memory_percent) if memory_percent else 0.0
        except ValueError:
            cpu = 0.0
            memory = 0.0
            
        return {
            'cpu_percent': cpu,
            'memory_percent': memory,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_gpu_usage(self) -> List[Dict]:
        """获取GPU使用情况"""
        cmd = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,utilization.memory --format=csv,noheader,nounits 2>/dev/null"
        output, code = self.execute_remote(cmd)
        
        gpus = []
        if code == 0 and output:
            for line in output.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'gpu_util': float(parts[2]),
                        'memory_used_mb': float(parts[3]),
                        'memory_total_mb': float(parts[4]),
                        'memory_util': float(parts[5]),
                        'memory_percent': (float(parts[3]) / float(parts[4])) * 100 if float(parts[4]) > 0 else 0
                    })
        return gpus
    
    def get_training_processes(self) -> List[Dict]:
        """获取训练相关进程"""
        # 查找所有Python训练进程
        cmd = "ps aux | grep -E 'train|python.*train' | grep -v grep | awk '{print $2, $3, $4, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20}'"
        output, code = self.execute_remote(cmd)
        
        processes = []
        if code == 0 and output:
            for line in output.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 4:
                    pid = parts[0]
                    cpu = float(parts[1]) if parts[1] else 0.0
                    mem = float(parts[2]) if parts[2] else 0.0
                    cmd_line = ' '.join(parts[3:])
                    
                    # 检查是否使用GPU
                    uses_gpu = False
                    gpu_check_cmd = f"nvidia-smi pmon -c 1 -s mu 2>/dev/null | grep {pid} || echo ''"
                    gpu_output, _ = self.execute_remote(gpu_check_cmd)
                    uses_gpu = bool(gpu_output.strip())
                    
                    processes.append({
                        'pid': pid,
                        'cpu_percent': cpu,
                        'memory_percent': mem,
                        'command': cmd_line,
                        'uses_gpu': uses_gpu
                    })
        
        return processes
    
    def check_and_alert(self, resources: Dict):
        """检查资源使用并发送预警"""
        cpu = resources['cpu_percent']
        memory = resources['memory_percent']
        now = datetime.now()
        
        # CPU预警
        if cpu > THRESHOLDS['cpu_percent']:
            if self.high_cpu_start is None:
                self.high_cpu_start = now
            else:
                duration = (now - self.high_cpu_start).total_seconds()
                if duration > THRESHOLDS['cpu_warn_duration']:
                    self.send_alert('CPU', cpu, duration)
        else:
            self.high_cpu_start = None
        
        # 内存预警
        if memory > THRESHOLDS['memory_percent']:
            if self.high_memory_start is None:
                self.high_memory_start = now
            else:
                duration = (now - self.high_memory_start).total_seconds()
                if duration > THRESHOLDS['memory_warn_duration']:
                    self.send_alert('Memory', memory, duration)
        else:
            self.high_memory_start = None
    
    def send_alert(self, resource_type: str, usage: float, duration: float):
        """发送预警"""
        if self.alert_sent:
            return
        
        message = f"""
警告：服务器资源使用过高！

资源类型: {resource_type}
当前使用率: {usage:.1f}%
高负载持续时间: {duration:.0f}秒
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

请检查：
1. 是否有异常进程占用资源
2. 训练任务是否正确使用GPU
3. 是否需要清理或重启服务

服务器: {self.server_user}@{self.server_host}:{self.server_port}
"""
        
        logger.warning(message)
        self.alert_sent = True
        
        # TODO: 可以添加邮件通知等功能
    
    def check_gpu_usage(self, processes: List[Dict]):
        """检查训练进程是否使用GPU"""
        warnings = []
        for proc in processes:
            # 检查是否是训练进程
            is_training = any(keyword in proc['command'].lower() for keyword in ['train.py', 'train'])
            
            if is_training and not proc['uses_gpu']:
                # 检查是否在CPU上运行训练（应该使用GPU）
                if not any(pattern in proc['command'].lower() for pattern in CPU_ONLY_PATTERNS):
                    warnings.append({
                        'pid': proc['pid'],
                        'command': proc['command'],
                        'cpu_percent': proc['cpu_percent'],
                        'memory_percent': proc['memory_percent'],
                        'message': '训练进程未使用GPU，可能占用CPU资源'
                    })
        
        return warnings
    
    def auto_stop_tasks(self, resources: Dict, processes: List[Dict]):
        """自动停止任务（当资源使用超过90%时）"""
        cpu = resources['cpu_percent']
        memory = resources['memory_percent']
        now = datetime.now()
        
        # 检查是否需要停止
        should_stop = False
        stop_reason = ""
        
        if cpu > CLEANUP_THRESHOLDS['cpu_percent']:
            if self.high_cpu_start is None:
                self.high_cpu_start = now
            else:
                duration = (now - self.high_cpu_start).total_seconds()
                if duration > THRESHOLDS['auto_stop_duration']:
                    should_stop = True
                    stop_reason = f"CPU使用率{cpu:.1f}%超过90%持续{duration:.0f}秒"
        else:
            self.high_cpu_start = None
        
        if memory > CLEANUP_THRESHOLDS['memory_percent']:
            if self.high_memory_start is None:
                self.high_memory_start = now
            else:
                duration = (now - self.high_memory_start).total_seconds()
                if duration > THRESHOLDS['auto_stop_duration']:
                    should_stop = True
                    stop_reason = f"内存使用率{memory:.1f}%超过90%持续{duration:.0f}秒"
        else:
            self.high_memory_start = None
        
        if should_stop and not self.stop_triggered and THRESHOLDS['auto_stop']:
            logger.critical(f"⚠️  自动停止任务: {stop_reason}")
            self.stop_triggered = True
            
            # 停止所有训练进程
            training_pids = []
            for proc in processes:
                is_training = any(keyword in proc['command'].lower() for keyword in ['train.py', 'train', 'python.*train'])
                if is_training:
                    training_pids.append(proc['pid'])
            
            if training_pids:
                for pid in training_pids:
                    logger.critical(f"停止训练进程 PID={pid}")
                    self.execute_remote(f"kill -15 {pid}")  # 先尝试SIGTERM
                    time.sleep(2)
                    # 如果还在运行，强制kill
                    check_cmd = f"ps -p {pid} > /dev/null 2>&1 && kill -9 {pid} || echo '已停止'"
                    self.execute_remote(check_cmd)
            
            # 停止所有高资源占用的非GPU进程
            for proc in processes:
                if proc['cpu_percent'] > 50 and not proc['uses_gpu']:
                    # 排除系统进程
                    if any(pattern in proc['command'].lower() for pattern in ['jupyter', 'notebook', 'ssh', 'systemd', 'kernel']):
                        continue
                    logger.critical(f"停止高资源占用进程 PID={proc['pid']}, CPU={proc['cpu_percent']:.1f}%")
                    self.execute_remote(f"kill -15 {proc['pid']}")
    
    def cleanup_abnormal_processes(self, resources: Dict, processes: List[Dict]):
        """清理异常进程（已废弃，使用auto_stop_tasks代替）"""
        pass
    
    def generate_report(self, resources: Dict, gpus: List[Dict], processes: List[Dict]) -> str:
        """生成资源报告"""
        report = f"""
============================================================
服务器资源监控报告
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
服务器: {self.server_user}@{self.server_host}:{self.server_port}
============================================================

【系统资源】
CPU使用率: {resources['cpu_percent']:.1f}%
内存使用率: {resources['memory_percent']:.1f}%

【GPU资源】
"""
        for gpu in gpus:
            report += f"GPU {gpu['index']} ({gpu['name']}):\n"
            report += f"  GPU利用率: {gpu['gpu_util']:.1f}%\n"
            report += f"  显存使用: {gpu['memory_used_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB ({gpu['memory_percent']:.1f}%)\n"
        
        if processes:
            report += f"\n【训练进程】\n"
            for proc in processes:
                gpu_status = "✅ 使用GPU" if proc['uses_gpu'] else "❌ 未使用GPU"
                report += f"PID {proc['pid']}: {proc['command'][:50]}...\n"
                report += f"  CPU: {proc['cpu_percent']:.1f}%, MEM: {proc['memory_percent']:.1f}%, {gpu_status}\n"
        else:
            report += "\n【训练进程】\n  未发现训练进程\n"
        
        return report
    
    def run_monitor(self, interval: int = 60, max_iterations: Optional[int] = None):
        """运行监控循环"""
        logger.info(f"开始监控服务器资源 (检查间隔: {interval}秒)")
        iteration = 0
        
        try:
            while True:
                iteration += 1
                if max_iterations and iteration > max_iterations:
                    break
                
                # 获取资源信息
                resources = self.get_cpu_memory_usage()
                gpus = self.get_gpu_usage()
                processes = self.get_training_processes()
                
                # 生成报告
                report = self.generate_report(resources, gpus, processes)
                logger.info(report)
                
                # 检查预警
                self.check_and_alert(resources)
                
                # 检查GPU使用
                gpu_warnings = self.check_gpu_usage(processes)
                if gpu_warnings:
                    for warn in gpu_warnings:
                        logger.warning(f"⚠️  {warn['message']}: PID={warn['pid']}, CMD={warn['command']}")
                
                # 自动停止任务（如果资源使用过高）
                self.auto_stop_tasks(resources, processes)
                
                # 保存报告到文件
                report_file = Path('resource_monitor_report.txt')
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                # 等待下次检查
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("监控已停止")
        except Exception as e:
            logger.error(f"监控出错: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description='服务器资源监控和预警系统')
    parser.add_argument('--host', type=str, default='10.103.92.120', help='服务器地址')
    parser.add_argument('--port', type=int, default=1066, help='SSH端口')
    parser.add_argument('--user', type=str, default='jiangh', help='用户名')
    parser.add_argument('--interval', type=int, default=30, help='检查间隔（秒）')
    parser.add_argument('--once', action='store_true', help='只运行一次检查')
    parser.add_argument('--monitor-pids', type=str, help='要监控的进程PID列表（逗号分隔）')
    
    args = parser.parse_args()
    
    monitor = ResourceMonitor(
        server_host=args.host,
        server_port=args.port,
        server_user=args.user
    )
    
    if args.monitor_pids:
        monitor.monitored_pids = [int(pid.strip()) for pid in args.monitor_pids.split(',')]
    
    if args.once:
        # 只运行一次检查
        resources = monitor.get_cpu_memory_usage()
        gpus = monitor.get_gpu_usage()
        processes = monitor.get_training_processes()
        report = monitor.generate_report(resources, gpus, processes)
        print(report)
        monitor.check_and_alert(resources)
    else:
        # 持续监控
        monitor.run_monitor(interval=args.interval)


if __name__ == '__main__':
    main()
