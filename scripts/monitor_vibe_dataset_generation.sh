#!/bin/bash
# Vibe Search数据集生成监控脚本

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/generate_vibe_dataset.log"
PID_FILE="$PROJECT_ROOT/generate_vibe_dataset.pid"

echo "============================================================"
echo "Vibe Search数据集生成监控"
echo "============================================================"

# 检查进程是否运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 进程运行中（PID: $PID）"
    else
        echo "❌ 进程已停止（PID文件存在但进程不存在）"
        rm -f "$PID_FILE"
    fi
else
    echo "⚠️  未找到PID文件，进程可能未启动"
fi

echo ""

# 显示最新日志
if [ -f "$LOG_FILE" ]; then
    echo "📊 最新进度（最后30行）:"
    echo "------------------------------------------------------------"
    tail -30 "$LOG_FILE" | grep -E "(处理chunk|情绪评分|查询生成|成功生成|已达到目标|处理完成|数据集统计|数据集生成完成)"
    echo "------------------------------------------------------------"
    
    # 统计信息
    TOTAL_PROCESSED=$(grep -c "处理chunk:" "$LOG_FILE" 2>/dev/null || echo "0")
    SUCCESSFUL=$(grep -c "成功生成查询" "$LOG_FILE" 2>/dev/null || echo "0")
    HIGH_EMOTION=$(grep -c "情绪密度足够" "$LOG_FILE" 2>/dev/null || echo "0")
    
    echo ""
    echo "📈 统计信息:"
    echo "  总处理: $TOTAL_PROCESSED 个chunks"
    echo "  高情绪密度: $HIGH_EMOTION 个"
    echo "  成功生成查询: $SUCCESSFUL 个"
else
    echo "⚠️  日志文件不存在，请等待几秒..."
fi

echo ""
echo "实时监控:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看完整日志:"
echo "  cat $LOG_FILE"
echo "============================================================"
