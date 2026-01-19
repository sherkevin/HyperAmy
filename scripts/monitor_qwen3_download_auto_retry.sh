#!/bin/bash
# Qwen3-Embedding-8B 自动监控和重试下载脚本
# 如果下载停止，自动重启

LOG_FILE="monitor_qwen3_download_$(date +%Y%m%d_%H%M%S).log"
MODEL_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B"
TARGET_SIZE=$((16 * 1024 * 1024 * 1024))  # 16GB in bytes
CHECK_INTERVAL=300  # 每5分钟检查一次
MAX_RETRIES=100  # 最大重试次数

echo "============================================================" | tee -a "$LOG_FILE"
echo "Qwen3-Embedding-8B 自动监控和重试下载" | tee -a "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    current_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "" | tee -a "$LOG_FILE"
    echo "[$current_time] 检查下载状态..." | tee -a "$LOG_FILE"
    
    # 检查当前大小
    if [ -d "$MODEL_DIR" ]; then
        current_size=$(du -sb "$MODEL_DIR" 2>/dev/null | awk '{print $1}')
        current_size_gb=$(python3 -c "print(f'{int($current_size) / (1024**3):.2f}')")
        progress=$(python3 -c "print(f'{int($current_size) * 100 / $TARGET_SIZE:.1f}')")
        
        echo "  当前大小: ${current_size_gb} GB (${progress}%)" | tee -a "$LOG_FILE"
        
        # 检查是否完成
        if [ "$current_size" -ge $((TARGET_SIZE - 1024*1024*1024)) ]; then  # 允许1GB误差
            echo "✅ 下载完成！" | tee -a "$LOG_FILE"
            echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
            exit 0
        fi
    else
        current_size=0
        current_size_gb="0.00"
        progress="0.0"
        echo "  模型目录不存在，开始首次下载" | tee -a "$LOG_FILE"
    fi
    
    # 检查下载进程
    download_pid=$(pgrep -f "download_qwen3_nohup\|python.*Qwen3-Embedding" | head -1)
    
    if [ -n "$download_pid" ]; then
        # 检查进程是否真的在运行
        if ps -p "$download_pid" > /dev/null 2>&1; then
            # 检查进程是否在下载（通过文件大小变化）
            before_size=$current_size
            sleep 30
            after_size=$(du -sb "$MODEL_DIR" 2>/dev/null | awk '{print $1}')
            size_diff=$((after_size - before_size))
            
            if [ "$size_diff" -gt 1048576 ]; then  # 至少1MB变化
                speed_mb=$(python3 -c "print(f'{int($size_diff) / (1024**2) / 30:.2f}')")
                echo "  ✅ 下载进行中 (PID: $download_pid, 速度: ${speed_mb} MB/s)" | tee -a "$LOG_FILE"
                retry_count=0  # 重置重试计数
            else
                echo "  ⚠️  进程存在但下载速度太慢，准备重启..." | tee -a "$LOG_FILE"
                kill "$download_pid" 2>/dev/null
                sleep 2
                download_pid=""
            fi
        else
            echo "  ⚠️  进程已停止，准备重启..." | tee -a "$LOG_FILE"
            download_pid=""
        fi
    else
        echo "  ⚠️  未找到下载进程，启动下载..." | tee -a "$LOG_FILE"
    fi
    
    # 如果没有运行中的进程，启动下载
    if [ -z "$download_pid" ]; then
        retry_count=$((retry_count + 1))
        echo "  重试次数: $retry_count/$MAX_RETRIES" | tee -a "$LOG_FILE"
        
        download_log="download_qwen3_auto_retry_$(date +%Y%m%d_%H%M%S).log"
        
        nohup /opt/conda/envs/PyTorch-2.4.1/bin/python3 << 'PYEOF' > "$download_log" 2>&1 &
import os
import sys
import time
import signal

os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

print('='*70)
print('Qwen3-Embedding-8B 自动重试下载')
print('='*70)
print(f'开始时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')

def signal_handler(sig, frame):
    print('\n收到中断信号，保存进度...')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

try:
    from transformers import AutoModel, AutoTokenizer
    
    print('\n加载模型（支持断点续传）...')
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen3-Embedding-8B', 
        trust_remote_code=True,
        resume_download=True
    )
    
    model = AutoModel.from_pretrained(
        'Qwen/Qwen3-Embedding-8B', 
        trust_remote_code=True,
        resume_download=True
    )
    
    print('\n✅ 模型下载完成！')
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'参数量: {param_count:.2f}B')
    print(f'完成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    
except Exception as e:
    print(f'\n❌ 下载失败: {str(e)}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

        new_pid=$!
        sleep 5
        
        if ps -p "$new_pid" > /dev/null 2>&1; then
            echo "  ✅ 下载已重启 (PID: $new_pid, 日志: $download_log)" | tee -a "$LOG_FILE"
        else
            echo "  ❌ 启动失败，查看日志: $download_log" | tee -a "$LOG_FILE"
            tail -10 "$download_log" | tee -a "$LOG_FILE"
        fi
    fi
    
    # 等待检查间隔
    echo "  下次检查: ${CHECK_INTERVAL}秒后..." | tee -a "$LOG_FILE"
    sleep "$CHECK_INTERVAL"
done

echo "❌ 达到最大重试次数，停止监控" | tee -a "$LOG_FILE"
exit 1
