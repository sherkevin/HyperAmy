#!/bin/bash
# 监控7B模型训练进度

SERVER="jiangh@10.103.92.120"
PORT="1066"
REMOTE_EMOS_DIR="/public/jiangh/emos"

echo "============================================================"
echo "监控7B模型训练进度"
echo "============================================================"

ssh -p $PORT $SERVER << 'REMOTE_SCRIPT'
cd /public/jiangh/emos

echo "【1】训练进程状态："
if pgrep -f "train.py" > /dev/null; then
    echo "✅ 训练进程正在运行"
    ps aux | grep train.py | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "时间:", $10}'
else
    echo "❌ 训练进程未运行"
fi

echo ""
echo "【2】GPU使用情况："
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | head -1 | awk -F', ' '{print "  GPU使用率:", $1, "  显存:", $2 "/" $3, "  温度:", $4}'

echo ""
echo "【3】最新训练日志（最后15行）："
LATEST_LOG=$(ls -t logs/train_7b_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    tail -15 "$LATEST_LOG"
else
    echo "  日志文件未找到"
fi

echo ""
echo "【4】Checkpoint文件："
if [ -d "checkpoints" ]; then
    ls -lh checkpoints/*.pt 2>/dev/null | tail -3 | awk '{print "  " $9, "(" $5 ")"}'
    if [ $? -ne 0 ]; then
        echo "  暂无checkpoint文件"
    fi
else
    echo "  checkpoint目录不存在"
fi

echo ""
echo "============================================================"
echo "提示："
echo "  实时查看: tail -f $LATEST_LOG"
echo "  完整日志: cat $LATEST_LOG"
echo "============================================================"

REMOTE_SCRIPT
