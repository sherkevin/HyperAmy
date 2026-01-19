#!/bin/bash
# 监控Qwen3-Embedding-8B下载进度

SERVER="jiangh@10.103.92.120"
PORT="1066"

ssh -p $PORT $SERVER << 'REMOTE_SCRIPT'
cd /public/jiangh/emos

echo "============================================================"
echo "Qwen3-Embedding-8B 下载监控"
echo "============================================================"

# 检查下载进程
DOWNLOAD_PID=$(pgrep -f "download_qwen3" | head -1)
if [ -n "$DOWNLOAD_PID" ]; then
    echo "✅ 下载进程运行中 (PID: $DOWNLOAD_PID)"
    ps aux | grep $DOWNLOAD_PID | grep -v grep | awk '{print "  CPU:", $3"%", "MEM:", $4"%", "运行时间:", $10}'
else
    echo "❌ 下载进程未运行"
fi

# 检查日志
DOWNLOAD_LOG=$(ls -t download_qwen3_embedding_fast_*.log 2>/dev/null | head -1)
if [ -f "$DOWNLOAD_LOG" ]; then
    echo ""
    echo "【最新下载日志（最后20行）】"
    tail -20 "$DOWNLOAD_LOG"
fi

# 检查文件大小
echo ""
echo "【模型文件大小】"
MODEL_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B"
if [ -d "$MODEL_DIR" ]; then
    SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | awk '{print $1}')
    echo "  当前大小: $SIZE"
    
    # 检查权重文件
    WEIGHT_FILES=$(find "$MODEL_DIR" -name "*.safetensors" -o -name "*.bin" 2>/dev/null)
    if [ -n "$WEIGHT_FILES" ]; then
        echo "  权重文件:"
        echo "$WEIGHT_FILES" | while read file; do
            if [ -f "$file" ]; then
                FILE_SIZE=$(du -h "$file" 2>/dev/null | awk '{print $1}')
                FILE_NAME=$(basename "$file")
                echo "    $FILE_NAME: $FILE_SIZE"
            fi
        done
        
        # 计算总大小
        TOTAL_SIZE=$(echo "$WEIGHT_FILES" | xargs du -ch 2>/dev/null | tail -1 | awk '{print $1}')
        echo "  总权重大小: $TOTAL_SIZE"
        
        # 估算进度（假设完整模型约16GB）
        TOTAL_BYTES=$(echo "$WEIGHT_FILES" | xargs du -cb 2>/dev/null | tail -1 | awk '{print $1}')
        EXPECTED_BYTES=$((16 * 1024 * 1024 * 1024))  # 16GB
        if [ "$TOTAL_BYTES" -gt 0 ] && [ "$EXPECTED_BYTES" -gt 0 ]; then
            PROGRESS=$(echo "scale=2; $TOTAL_BYTES * 100 / $EXPECTED_BYTES" | bc)
            echo "  下载进度: ${PROGRESS}%"
        fi
    else
        echo "  ⏳ 权重文件未找到，正在下载中..."
    fi
else
    echo "  ❌ 模型目录不存在"
fi

REMOTE_SCRIPT
