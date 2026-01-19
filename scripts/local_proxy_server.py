#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地代理服务器 - 将实验信息转发到云服务器
运行在本地，从hyperamy-server获取实验信息，然后通过HTTP API提供给云服务器
"""

import subprocess
import time
import re
import json
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域访问

# 服务器配置
SERVER = "hyperamy-server"
PROJECT_ROOT = "/public/jiangh/HyperAmy"
LOG_FILE = f"{PROJECT_ROOT}/test_three_methods_comparison_monte_cristo.log"
RESULT_FILE = f"{PROJECT_ROOT}/outputs/three_methods_comparison_monte_cristo/comparison_results.json"

def get_process_info(server=SERVER):
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

def get_latest_progress(server=SERVER):
    """获取最新进度"""
    cmd = f'ssh {server} "cd {PROJECT_ROOT} && tail -100 test_three_methods_comparison_monte_cristo.log 2>/dev/null"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if ('Extracting emotion vectors:' in line or 'NER:' in line or 'Extracting triples:' in line or 
                    'Processing' in line or 'Batch Encoding' in line or '处理chunks:' in line) and '%|' in line:
                    match = re.search(r'(\d+)%\|.*?(\d+)/(\d+).*?\[(.*?)<(.*?), (.*?)\]', line)
                    if match:
                        percent = int(match.group(1))
                        current = int(match.group(2))
                        total = int(match.group(3))
                        elapsed = match.group(4).strip()
                        remaining = match.group(5).strip()
                        speed = match.group(6).strip()
                        task_name = "未知任务"
                        if 'Extracting emotion vectors' in line or '提取情绪向量' in line:
                            task_name = "提取情绪向量"
                        elif 'NER:' in line:
                            task_name = "命名实体识别"
                        elif 'Extracting triples' in line or '提取三元组' in line:
                            task_name = "提取三元组"
                        elif 'Processing' in line or '处理' in line:
                            task_name = "处理文档"
                        elif 'Batch Encoding' in line:
                            task_name = "批量编码"
                        elif '处理chunks' in line:
                            task_name = "处理chunks"
                        return {
                            'found': True,
                            'percent': percent,
                            'current': current,
                            'total': total,
                            'elapsed': elapsed,
                            'remaining': remaining,
                            'speed': speed,
                            'task_name': task_name
                        }
    except:
        pass
    return {'found': False}

def get_steps_status(server=SERVER):
    """获取步骤状态"""
    cmd = f'ssh {server} "cd {PROJECT_ROOT} && grep -E \'【步骤|初始化成功|完成|索引完成|检索完成\' test_three_methods_comparison_monte_cristo.log 2>/dev/null | tail -20"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            steps = {f'步骤{i}': False for i in range(1, 11)}
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
                steps['步骤6'] = True
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

def get_result_file_status(server=SERVER):
    """检查结果文件状态"""
    cmd = f'ssh {server} "cd {PROJECT_ROOT} && test -f outputs/three_methods_comparison_monte_cristo/comparison_results.json && echo exists || echo not_exists"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if 'exists' in result.stdout:
            return True
    except:
        pass
    return False

def get_recent_logs(server=SERVER, num_lines=5):
    """获取最新日志"""
    cmd = f'ssh {server} "cd {PROJECT_ROOT} && tail -{num_lines} test_three_methods_comparison_monte_cristo.log 2>/dev/null"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return [line.strip()[:150] for line in lines if line.strip()]
    except:
        pass
    return []

@app.route('/api/progress', methods=['GET'])
def api_progress():
    """API: 获取实验进度（供云服务器调用）"""
    try:
        process_info = get_process_info()
        progress = get_latest_progress()
        steps = get_steps_status()
        result_exists = get_result_file_status()
        logs = get_recent_logs()
        
        status = 'waiting'
        status_text = '等待中'
        
        if result_exists:
            status = 'completed'
            status_text = '已完成'
        elif process_info.get('running'):
            if progress.get('found'):
                status = 'running'
                status_text = '运行中'
            else:
                status = 'running'
                status_text = '运行中（等待进度）'
        else:
            if progress.get('found') and progress.get('percent', 0) >= 100:
                status = 'completed'
                status_text = '已完成'
            else:
                status = 'waiting'
                status_text = '未运行'
        
        return jsonify({
            'status': status,
            'status_text': status_text,
            'process': process_info,
            'progress': progress,
            'steps': steps,
            'result_exists': result_exists,
            'logs': logs,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🔗 本地代理服务器（实验信息转发）")
    print("=" * 70)
    print("📡 API地址: http://0.0.0.0:8888/api/progress")
    print("=" * 70)
    print("💡 此服务从hyperamy-server获取实验信息，供云服务器调用")
    print("=" * 70)
    app.run(host='0.0.0.0', port=8888, debug=False, threaded=True)
