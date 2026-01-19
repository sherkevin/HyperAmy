#!/bin/bash
# 监控两个模型的训练进度

SERVER="jiangh@10.103.92.120"
PORT="1066"
REMOTE_EMOS_DIR="/public/jiangh/emos"

echo "============================================================"
echo "监控两个模型的训练进度"
echo "============================================================"

ssh -p $PORT $SERVER << 'REMOTE_SCRIPT'
cd /public/jiangh/emos

echo "【1】训练进程状态："
QWEN_PID=$(pgrep -f "train.py.*Qwen" | head -1)
ROBERTA_PID=$(pgrep -f "train.py.*roberta" | head -1)

if [ -n "$QWEN_PID" ]; then
    echo "✅ Qwen-7B: 运行中 (PID: $QWEN_PID)"
    ps aux | grep $QWEN_PID | grep -v grep | awk '{print "  CPU:", $3"%", "MEM:", $4"%", "时间:", $10}'
else
    echo "❌ Qwen-7B: 未运行"
fi

if [ -n "$ROBERTA_PID" ]; then
    echo "✅ RoBERTa-large: 运行中 (PID: $ROBERTA_PID)"
    ps aux | grep $ROBERTA_PID | grep -v grep | awk '{print "  CPU:", $3"%", "MEM:", $4"%", "时间:", $10}'
else
    echo "❌ RoBERTa-large: 未运行"
fi

echo ""
echo "【2】GPU使用情况："
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "【3】最新训练日志："

# Qwen-7B日志
QWEN_LOG=$(ls -t logs/train_qwen7b_*.log 2>/dev/null | head -1)
if [ -f "$QWEN_LOG" ]; then
    echo ""
    echo "--- Qwen-7B (最后10行) ---"
    tail -10 "$QWEN_LOG" | grep -E "Epoch|Loss|Step|Best|完成|complete" || tail -10 "$QWEN_LOG"
else
    echo "Qwen-7B日志未找到"
fi

# RoBERTa日志
ROBERTA_LOG=$(ls -t logs/train_roberta_large_*.log 2>/dev/null | head -1)
if [ -f "$ROBERTA_LOG" ]; then
    echo ""
    echo "--- RoBERTa-large (最后10行) ---"
    tail -10 "$ROBERTA_LOG" | grep -E "Epoch|Loss|Step|Best|完成|complete" || tail -10 "$ROBERTA_LOG"
else
    echo "RoBERTa日志未找到"
fi

echo ""
echo "【4】Checkpoint文件："
if [ -d "checkpoints/qwen7b" ]; then
    echo "Qwen-7B checkpoints:"
    ls -lh checkpoints/qwen7b/*.pt 2>/dev/null | tail -2 | awk '{print "  " $9, "(" $5 ")"}'
fi

if [ -d "checkpoints/roberta_large" ]; then
    echo "RoBERTa-large checkpoints:"
    ls -lh checkpoints/roberta_large/*.pt 2>/dev/null | tail -2 | awk '{print "  " $9, "(" $5 ")"}'
fi

REMOTE_SCRIPT
