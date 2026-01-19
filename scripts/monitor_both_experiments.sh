#!/bin/bash

# 同时监控两个实验的脚本

LOG_FILE_API="test_vibe_search_experiment_run.log"
LOG_FILE_EMOS="test_vibe_search_experiment_emos_gpu_run.log"
PID_FILE_API="test_vibe_search_experiment.pid"
PID_FILE_EMOS="test_vibe_search_experiment_emos_gpu.pid"

echo "🚀 启动双实验监控..."
echo "============================================================"
echo ""
echo "📊 实验1: LLM API版本"
echo "  日志文件: ${LOG_FILE_API}"
echo ""
echo "📊 实验2: Emos模型 + GPU版本"
echo "  日志文件: ${LOG_FILE_EMOS}"
echo ""
echo "============================================================"
echo ""
echo "实时监控（按 Ctrl+C 退出）..."
echo ""

# 监控函数
monitor_experiment() {
    LOG_FILE=$1
    EXPERIMENT_NAME=$2
    if [ -f "$LOG_FILE" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 ${EXPERIMENT_NAME}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 检查进程状态
        if ps aux | grep -E "$(basename $LOG_FILE)" | grep -v grep > /dev/null; then
            echo "✅ 状态: 运行中"
        else
            echo "⚠️  状态: 已完成或未启动"
        fi
        
        # 显示最后几行关键信息
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "最新进度:"
            tail -5 "$LOG_FILE" | grep -E "(提取情绪向量|Recall@1|实验完成|✅|GPU|MPS|步骤)" | tail -3 || tail -3 "$LOG_FILE"
        fi
        echo ""
    else
        echo "⚠️  日志文件不存在: ${LOG_FILE}"
        echo ""
    fi
}

# 循环监控
while true; do
    clear
    echo "🔄 双实验监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""
    
    monitor_experiment "$LOG_FILE_API" "实验1: LLM API版本"
    monitor_experiment "$LOG_FILE_EMOS" "实验2: Emos模型 + GPU版本"
    
    echo "============================================================"
    echo "按 Ctrl+C 退出监控"
    echo "刷新间隔: 5秒"
    
    sleep 5
done
