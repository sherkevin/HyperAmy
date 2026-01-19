#!/bin/bash
# GoT实验监控脚本

cd /public/jiangh/HyperAmy

echo "============================================================"
echo "GoT实验监控"
echo "============================================================"
echo ""

# 检查进程
if [ -f "got_experiment.pid" ]; then
    PID=$(cat got_experiment.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 实验进程运行中 (PID: $PID)"
        echo "   运行时间: $(ps -o etime= -p $PID 2>/dev/null | tr -d ' ')"
    else
        echo "❌ 实验进程已停止 (PID: $PID)"
    fi
else
    echo "⚠️ 未找到PID文件"
fi

echo ""

# 查找最新日志
LATEST_LOG=$(ls -t logs/got_experiment_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "最新日志: $LATEST_LOG"
    LOG_SIZE=$(wc -l < "$LATEST_LOG" 2>/dev/null || echo "0")
    echo "日志行数: $LOG_SIZE"
    echo ""
    
    echo "=== 日志尾部（最后30行） ==="
    tail -30 "$LATEST_LOG"
    echo ""
    
    # 统计信息
    echo "=== 统计信息 ==="
    ERROR_COUNT=$(grep -c "ERROR\|Traceback\|Failed" "$LATEST_LOG" 2>/dev/null || echo "0")
    BATCH_COUNT=$(grep -c "Batch Encoding" "$LATEST_LOG" 2>/dev/null || echo "0")
    COMPLETE_COUNT=$(grep -c "完成\|完成\|✅.*完成" "$LATEST_LOG" 2>/dev/null || echo "0")
    
    echo "  - 错误数: $ERROR_COUNT"
    echo "  - Batch Encoding进度: $BATCH_COUNT"
    echo "  - 完成标记: $COMPLETE_COUNT"
    
    # 检查是否完成
    if grep -q "实验完成\|所有方法完成\|评估完成" "$LATEST_LOG" 2>/dev/null; then
        echo ""
        echo "✅ 实验可能已完成"
    fi
else
    echo "⚠️ 未找到实验日志"
fi

echo ""
echo "=== GPU使用情况 ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "无法获取GPU信息"
