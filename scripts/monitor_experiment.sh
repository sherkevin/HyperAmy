#!/bin/bash
# 监控严谨对比实验

cd /media/data4/jiangh/Amygdala/hyperamy_source

PID_FILE="outputs/rigorous_experiment/experiment.pid"
LOG_FILE="outputs/rigorous_experiment/experiment.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "实验进程ID: $PID"
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 实验正在运行"
        echo ""
        echo "进程信息:"
        ps -p $PID -o pid,pcpu,pmem,etime,cmd
        echo ""
        echo "最近日志 (最后20行):"
        tail -20 "$LOG_FILE" 2>/dev/null || echo "日志文件不存在或为空"
        echo ""
        echo "日志文件大小:"
        ls -lh "$LOG_FILE" 2>/dev/null || echo "日志文件不存在"
    else
        echo "❌ 实验进程不存在（可能已完成或出错）"
        echo ""
        echo "最后日志:"
        tail -50 "$LOG_FILE" 2>/dev/null || echo "日志文件不存在"
    fi
else
    echo "❌ PID文件不存在"
fi


