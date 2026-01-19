#!/bin/bash
# 运行任务时自动启动资源监控
# 用法: ./run_task_with_monitor.sh <command>
# 例如: ./run_task_with_monitor.sh "python train.py --model_name Qwen3-Embedding-8B"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MONITOR_SCRIPT="$SCRIPT_DIR/resource_monitor_server.py"
MONITOR_PID_FILE="resource_monitor.pid"
MONITOR_LOG="logs/resource_monitor.log"

# 创建logs目录
mkdir -p logs

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <command>"
    echo "例如: $0 'python train.py --model_name Qwen3-Embedding-8B'"
    exit 1
fi

TASK_COMMAND="$@"

echo "============================================================"
echo "启动任务（带资源监控）"
echo "============================================================"
echo "任务命令: $TASK_COMMAND"
echo ""

# 停止旧的监控（如果存在）
if [ -f "$MONITOR_PID_FILE" ]; then
    OLD_PID=$(cat "$MONITOR_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "停止旧的监控进程 (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null
        sleep 2
    fi
    rm -f "$MONITOR_PID_FILE"
fi

# 启动资源监控
echo "启动资源监控系统..."
nohup python3 "$MONITOR_SCRIPT" \
    --host 10.103.92.120 \
    --port 1066 \
    --user jiangh \
    --interval 30 \
    > "$MONITOR_LOG" 2>&1 &

MONITOR_PID=$!
echo $MONITOR_PID > "$MONITOR_PID_FILE"

echo "✅ 资源监控已启动 (PID: $MONITOR_PID)"
echo "监控日志: $MONITOR_LOG"
echo ""

# 等待监控启动
sleep 3

# 定义清理函数
cleanup() {
    echo ""
    echo "============================================================"
    echo "清理资源..."
    echo "============================================================"
    
    # 停止监控
    if [ -f "$MONITOR_PID_FILE" ]; then
        MONITOR_PID=$(cat "$MONITOR_PID_FILE")
        if ps -p "$MONITOR_PID" > /dev/null 2>&1; then
            echo "停止资源监控 (PID: $MONITOR_PID)..."
            kill "$MONITOR_PID" 2>/dev/null
        fi
        rm -f "$MONITOR_PID_FILE"
    fi
    
    echo "✅ 清理完成"
    exit 0
}

# 注册清理函数
trap cleanup SIGINT SIGTERM EXIT

# 运行任务
echo "============================================================"
echo "开始执行任务..."
echo "============================================================"
echo ""

eval "$TASK_COMMAND"
TASK_EXIT_CODE=$?

echo ""
echo "============================================================"
echo "任务执行完成 (退出码: $TASK_EXIT_CODE)"
echo "============================================================"

# 清理会在trap中自动执行
exit $TASK_EXIT_CODE
