#!/bin/bash

# 监控《基督山伯爵》对比实验进度

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PID_FILE="test_monte_cristo_comparison.pid"
LOG_FILE="test_monte_cristo_comparison.log"
RESULT_FILE="results/monte_cristo_comparison.json"

echo "======================================================================"
echo "《基督山伯爵》对比实验监控"
echo "======================================================================"

# 1. 检查进程状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "【1】进程状态: ✅ 运行中 (PID: $PID)"
        ps -p $PID -o pid,etime,%cpu,%mem,command | tail -1
    else
        echo "【1】进程状态: ❌ 进程已结束"
    fi
else
    echo "【1】进程状态: ⚠️  PID文件不存在"
fi

echo ""

# 2. 显示最新日志
echo "【2】最新日志（最后15行）:"
if [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE"
else
    echo "  ⏳ 日志文件尚未创建"
fi

echo ""

# 3. 检查结果文件状态
echo "【3】结果文件状态:"
if [ -f "$RESULT_FILE" ]; then
    echo "  ✅ 结果文件存在"
    TOTAL_QA=$(python3 -c "import json; f=open('$RESULT_FILE'); data=json.load(f); print(len(data))" 2>/dev/null || echo "0")
    echo "  已处理QA对: $TOTAL_QA"
    ls -lh "$RESULT_FILE"
else
    echo "  ⏳ 结果文件尚未创建"
fi

echo ""
echo "======================================================================"
echo "持续监控命令:"
echo "  watch -n 10 './scripts/monitor_monte_cristo_experiment.sh'"
echo "  或: tail -f $LOG_FILE"
echo "======================================================================"

