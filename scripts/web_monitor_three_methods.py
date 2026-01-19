#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三种方法对比实验 - Web实时监控服务器（手机友好版）
提供响应式Web界面和API接口来实时监控实验进度
"""

import subprocess
import time
import re
import json
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# 服务器配置
# 实验服务器的直接API（如果可访问）
EXPERIMENT_SERVER_API = "http://10.103.92.120:8888/api/results"  # 实验服务器的结果API
EXPERIMENT_SERVER_STATUS = "http://10.103.92.120:8888/api/status"  # 实验服务器的状态API
USE_DIRECT_API = True  # 如果实验服务器可直接访问，设置为True

# 优先使用本地代理服务器（通过SSH反向隧道，云服务器访问localhost:8888）
LOCAL_PROXY_URL = "http://localhost:9999/api/progress"  # 通过SSH反向隧道访问本地代理（端口9999）
USE_LOCAL_PROXY = False  # 如果本地代理不可用，设置为False

# 直接SSH配置（备用方案）
SERVER = "hyperamy-server"
PROJECT_ROOT = "/public/jiangh/HyperAmy"
LOG_FILE = f"{PROJECT_ROOT}/test_three_methods_comparison_monte_cristo.log"
RESULT_FILE = f"{PROJECT_ROOT}/outputs/three_methods_comparison_monte_cristo/comparison_results.json"

# HTML模板（手机友好，响应式设计）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>实验进度监控</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 10px;
            color: #333;
        }
        
        .container {
            max-width: 100%;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 20px;
            margin-bottom: 10px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .header h1 {
            font-size: 24px;
            color: #333;
            margin-bottom: 5px;
        }
        
        .header .subtitle {
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }
        
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-top: 5px;
        }
        
        .status-running { background: #28a745; color: white; }
        .status-waiting { background: #ffc107; color: #333; }
        .status-completed { background: #007bff; color: white; }
        
        .progress-section {
            margin: 20px 0;
        }
        
        .progress-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 14px;
        }
        
        .progress-label {
            font-weight: bold;
            color: #666;
        }
        
        .progress-value {
            font-weight: bold;
            color: #333;
            font-size: 16px;
        }
        
        .progress-bar-container {
            background: #e9ecef;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            position: relative;
            margin: 10px 0;
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 15px 0;
        }
        
        .stat-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        
        .stat-label {
            font-size: 11px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
        
        .steps-section {
            margin: 20px 0;
        }
        
        .steps-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        
        .step-item {
            display: flex;
            align-items: center;
            padding: 10px;
            margin-bottom: 8px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 13px;
        }
        
        .step-icon {
            font-size: 18px;
            margin-right: 10px;
            width: 24px;
            text-align: center;
        }
        
        .step-text {
            flex: 1;
        }
        
        .log-section {
            margin: 20px 0;
        }
        
        .log-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        
        .log-container {
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 10px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            line-height: 1.5;
        }
        
        .log-line {
            margin: 2px 0;
            white-space: pre-wrap;
            word-break: break-all;
        }
        
        .update-info {
            text-align: center;
            font-size: 11px;
            color: #999;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #f0f0f0;
        }
        
        .error-message {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 13px;
        }
        
        /* 加载动画 */
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .loading::after {
            content: "...";
            animation: dots 1.5s steps(4, end) infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: "."; }
            40% { content: ".."; }
            60%, 100% { content: "..."; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 实验进度监控</h1>
            <div class="subtitle">三种方法对比实验</div>
            <div id="status-badge" class="status-badge status-waiting">等待中</div>
        </div>
        
        <div id="content">
            <div class="loading">正在加载数据</div>
        </div>
        
        <div class="update-info">
            <div>⏱️ 每3秒自动刷新</div>
            <div id="last-update" style="margin-top: 5px;">--</div>
        </div>
    </div>
    
    <script>
        let updateInterval;
        
        function formatTime(timeStr) {
            if (!timeStr || timeStr === 'N/A') return 'N/A';
            return timeStr;
        }
        
        function updateProgress() {
            fetch('/api/progress')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('content').innerHTML = 
                            `<div class="error-message">❌ ${data.error}</div>`;
                        return;
                    }
                    
                    // 更新状态徽章
                    const statusBadge = document.getElementById('status-badge');
                    const status = data.status || 'waiting';
                    statusBadge.textContent = data.status_text || '等待中';
                    statusBadge.className = 'status-badge status-' + status;
                    
                    // 更新最后更新时间
                    document.getElementById('last-update').textContent = 
                        `最后更新: ${new Date().toLocaleTimeString('zh-CN')}`;
                    
                    // 构建内容HTML
                    let html = '';
                    
                    // 进度信息
                    if (data.progress && data.progress.found) {
                        const p = data.progress;
                        const percent = p.percent || 0;
                        html += `
                            <div class="progress-section">
                                <div class="progress-info">
                                    <span class="progress-label">📈 ${p.task_name || '当前任务'}</span>
                                    <span class="progress-value">${percent}%</span>
                                </div>
                                <div class="progress-bar-container">
                                    <div class="progress-bar" style="width: ${percent}%">
                                        ${percent >= 10 ? percent + '%' : ''}
                                    </div>
                                </div>
                                <div class="stats-grid">
                                    <div class="stat-card">
                                        <div class="stat-label">已完成</div>
                                        <div class="stat-value">${p.current || 0}</div>
                                    </div>
                                    <div class="stat-card">
                                        <div class="stat-label">总数</div>
                                        <div class="stat-value">${p.total || 0}</div>
                                    </div>
                                    <div class="stat-card">
                                        <div class="stat-label">已用时间</div>
                                        <div class="stat-value" style="font-size: 14px;">${formatTime(p.elapsed || 'N/A')}</div>
                                    </div>
                                    <div class="stat-card">
                                        <div class="stat-label">预计剩余</div>
                                        <div class="stat-value" style="font-size: 14px;">${formatTime(p.remaining || 'N/A')}</div>
                                    </div>
                                </div>
                                <div style="margin-top: 10px; font-size: 12px; color: #666; text-align: center;">
                                    速度: ${p.speed || 'N/A'}
                                </div>
                            </div>
                        `;
                    } else {
                        html += `
                            <div class="progress-section">
                                <div class="loading">等待进度信息</div>
                            </div>
                        `;
                    }
                    
                    // 步骤状态
                    if (data.steps) {
                        html += `
                            <div class="steps-section">
                                <div class="steps-title">📋 实验步骤</div>
                        `;
                        
                        const stepNames = {
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
                        };
                        
                        const stepOrder = ['步骤1', '步骤2', '步骤3', '步骤4', '步骤5', 
                                         '步骤6', '步骤7', '步骤8', '步骤9', '步骤10'];
                        
                        stepOrder.forEach(stepKey => {
                            const completed = data.steps[stepKey] || false;
                            const stepName = stepNames[stepKey] || stepKey;
                            const icon = completed ? '✅' : '⏸️';
                            html += `
                                <div class="step-item">
                                    <span class="step-icon">${icon}</span>
                                    <span class="step-text">${stepKey}: ${stepName}</span>
                                </div>
                            `;
                        });
                        
                        html += `</div>`;
                    }
                    
                    // 进程信息
                    if (data.process && data.process.running) {
                        html += `
                            <div class="stats-grid" style="margin-top: 15px;">
                                <div class="stat-card">
                                    <div class="stat-label">进程ID</div>
                                    <div class="stat-value" style="font-size: 14px;">${data.process.pid || 'N/A'}</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">CPU使用</div>
                                    <div class="stat-value" style="font-size: 14px;">${data.process.cpu || '0'}%</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">内存使用</div>
                                    <div class="stat-value" style="font-size: 14px;">${data.process.mem || '0'}%</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">运行时间</div>
                                    <div class="stat-value" style="font-size: 12px;">${data.process.time || 'N/A'}</div>
                                </div>
                            </div>
                        `;
                    }
                    
                    // 结果文件状态
                    if (data.result_exists) {
                        html += `
                            <div style="margin-top: 15px; padding: 15px; background: #d4edda; border-radius: 10px; text-align: center; color: #155724;">
                                ✅ 结果文件已生成
                            </div>
                        `;
                    }
                    
                    // 最新日志
                    if (data.logs && data.logs.length > 0) {
                        html += `
                            <div class="log-section">
                                <div class="log-title">📝 最新日志</div>
                                <div class="log-container">
                        `;
                        data.logs.slice(-5).forEach(line => {
                            const escaped = line.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                            html += `<div class="log-line">${escaped}</div>`;
                        });
                        html += `
                                </div>
                            </div>
                        `;
                    }
                    
                    document.getElementById('content').innerHTML = html;
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('content').innerHTML = 
                        `<div class="error-message">❌ 连接失败: ${error.message}</div>`;
                });
        }
        
        // 每3秒自动刷新
        updateInterval = setInterval(updateProgress, 3000);
        updateProgress(); // 立即执行一次
        
        // 页面可见性改变时继续/暂停刷新
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                clearInterval(updateInterval);
            } else {
                updateInterval = setInterval(updateProgress, 3000);
                updateProgress();
            }
        });
    </script>
</body>
</html>
"""

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
    except Exception as e:
        pass
    return {'running': False}

def get_latest_progress(server=None):
    """获取最新进度"""
    cmd = f'ssh {server} "cd {PROJECT_ROOT} && tail -100 test_three_methods_comparison_monte_cristo.log 2>/dev/null"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                # 支持多种进度条格式
                if ('Extracting emotion vectors:' in line or 'NER:' in line or 'Extracting triples:' in line or 
                    'Processing' in line or 'Batch Encoding' in line or '处理chunks:' in line) and '%|' in line:
                    # 解析进度条: 72%|███████▏  | 7058/9735 [59:21<34:45,  1.28it/s]
                    match = re.search(r'(\d+)%\|.*?(\d+)/(\d+).*?\[(.*?)<(.*?), (.*?)\]', line)
                    if match:
                        percent = int(match.group(1))
                        current = int(match.group(2))
                        total = int(match.group(3))
                        elapsed = match.group(4).strip()
                        remaining = match.group(5).strip()
                        speed = match.group(6).strip()
                        # 提取任务名称
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
    except Exception as e:
        pass
    return {'found': False}

def get_steps_status(server=None):
    """获取步骤状态"""
    cmd = f'ssh {server} "cd {PROJECT_ROOT} && grep -E \'【步骤|初始化成功|完成|索引完成|检索完成\' test_three_methods_comparison_monte_cristo.log 2>/dev/null | tail -20"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            steps = {
                '步骤1': False, '步骤2': False, '步骤3': False, '步骤4': False,
                '步骤5': False, '步骤6': False, '步骤7': False, '步骤8': False,
                '步骤9': False, '步骤10': False
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
    except Exception as e:
        pass
    return {}

def get_result_file_status(server=SERVER):
    """检查结果文件状态"""
    if RUN_LOCAL:
        result_file = Path(RESULT_FILE)
        return result_file.exists()
    else:
        cmd = f'ssh {server} "cd {PROJECT_ROOT} && test -f outputs/three_methods_comparison_monte_cristo/comparison_results.json && echo exists || echo not_exists"'
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            if 'exists' in result.stdout:
                return True
        except:
            pass
        return False

def get_recent_logs(server=None, num_lines=5):
    """获取最新日志"""
    cmd = f'ssh {server} "cd {PROJECT_ROOT} && tail -{num_lines} test_three_methods_comparison_monte_cristo.log 2>/dev/null"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return [line.strip()[:150] for line in lines if line.strip()]  # 限制每行长度
    except:
        pass
    return []

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

def get_progress_from_local_proxy():
    """从本地代理服务器获取实验进度"""
    import requests
    try:
        response = requests.get(LOCAL_PROXY_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return None

@app.route('/api/progress')
def api_progress():
    """API: 获取实验进度"""
    try:
        # 优先从本地代理服务器获取数据
        if USE_LOCAL_PROXY:
            proxy_data = get_progress_from_local_proxy()
            if proxy_data:
                return jsonify(proxy_data)
        
        # 备用方案：直接SSH到实验服务器
        process_info = get_process_info()
        progress = get_latest_progress()
        steps = get_steps_status()
        result_exists = get_result_file_status()
        logs = get_recent_logs()
        
        # 确定状态
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
    import sys
    
    # 尝试使用5000端口，如果被占用则使用5001
    port = 5000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    print("=" * 70)
    print("🌐 三种方法对比实验 - Web实时监控服务器（手机版）")
    print("=" * 70)
    print(f"📱 访问地址: http://0.0.0.0:{port}")
    print(f"📱 手机访问: http://<你的IP地址>:{port}")
    print(f"📊 API地址: http://0.0.0.0:{port}/api/progress")
    print("=" * 70)
    print("💡 提示: 在同一网络下，手机浏览器可以访问此页面")
    print("=" * 70)
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"⚠️  端口 {port} 被占用，尝试使用 5001...")
            app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
        else:
            raise
