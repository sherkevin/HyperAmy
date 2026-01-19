#!/bin/bash
# 监控Fusion策略网格搜索实验进度

PROJECT_ROOT="/public/jiangh/HyperAmy"
PID_FILE="${PROJECT_ROOT}/fusion_grid_search.pid"
LOG_FILE="${PROJECT_ROOT}/fusion_strategy_grid_search.log"
PROGRESS_FILE="${PROJECT_ROOT}/outputs/fusion_strategy_grid_search/progress.json"
RESULTS_DIR="${PROJECT_ROOT}/outputs/fusion_strategy_grid_search/results"

echo "================================================================================"
echo "📊 Fusion 策略网格搜索实验监控"
echo "================================================================================"
echo ""

# 检查进程状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 实验进程正在运行"
        echo "   PID: $PID"
        ps -p $PID -o pid,etime,%cpu,%mem,cmd --no-headers | awk '{print "   运行时间: "$2", CPU: "$3", 内存: "$4}'
    else
        echo "❌ 进程不存在（可能已结束）"
        rm "$PID_FILE" 2>/dev/null
    fi
else
    echo "⚠️  PID文件不存在，实验可能未启动"
fi

echo ""

# 检查日志文件
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    LAST_UPDATE=$(stat -c %Y "$LOG_FILE" 2>/dev/null || stat -f %m "$LOG_FILE" 2>/dev/null)
    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - LAST_UPDATE))
    
    echo "📝 日志文件状态:"
    echo "   文件大小: $LOG_SIZE"
    if [ "$TIME_DIFF" -lt 300 ]; then
        echo "   ✅ 正在更新 (最近更新于 ${TIME_DIFF} 秒前)"
    else
        echo "   ⚠️  长时间未更新 (最近更新于 ${TIME_DIFF} 秒前)"
    fi
    echo ""
    echo "   最后10行日志:"
    tail -n 10 "$LOG_FILE" | sed 's/^/   /'
else
    echo "⚠️  日志文件不存在: $LOG_FILE"
fi

echo ""

# 检查进度
if [ -f "$PROGRESS_FILE" ]; then
    echo "📈 实验进度:"
    COMPLETED=$(python3 -c "import json; p=json.load(open('$PROGRESS_FILE')); print(len(p.get('completed_configs', [])))" 2>/dev/null || echo "0")
    FAILED=$(python3 -c "import json; p=json.load(open('$PROGRESS_FILE')); print(len(p.get('failed_configs', [])))" 2>/dev/null || echo "0")
    TOTAL=80
    
    COMPLETED_PERCENT=$((COMPLETED * 100 / TOTAL))
    REMAINING=$((TOTAL - COMPLETED))
    
    echo "   已完成: $COMPLETED / $TOTAL ($COMPLETED_PERCENT%)"
    echo "   失败: $FAILED"
    echo "   剩余: $REMAINING"
    echo ""
    
    # 进度条
    BAR_WIDTH=50
    FILLED=$((COMPLETED * BAR_WIDTH / TOTAL))
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')
    EMPTY=$(printf "%$((BAR_WIDTH - FILLED))s" | tr ' ' '░')
    echo "   进度: [$BAR$EMPTY] $COMPLETED_PERCENT%"
else
    echo "⚠️  进度文件不存在: $PROGRESS_FILE"
    echo "   实验可能刚刚开始，或进度文件尚未生成"
fi

echo ""

# 检查结果文件
if [ -d "$RESULTS_DIR" ]; then
    RESULT_COUNT=$(find "$RESULTS_DIR" -name "result_*.json" 2>/dev/null | wc -l)
    echo "📁 结果文件:"
    echo "   已生成结果数: $RESULT_COUNT"
    
    if [ "$RESULT_COUNT" -gt 0 ]; then
        # 找出最佳配置（Recall@10最高）
        echo ""
        echo "🏆 当前最佳配置（基于Recall@10）:"
        python3 << 'PYEOF'
import json
import glob
import os
import sys

results_dir = sys.argv[1]
result_files = glob.glob(os.path.join(results_dir, "result_*.json"))

best_recall_10 = None
best_config = None

for f in result_files:
    try:
        with open(f, 'r') as file:
            result = json.load(file)
            if 'error' not in result and 'metrics' in result:
                recall_10 = result['metrics'].get('recall_at_k', {}).get(10, 0.0)
                if best_recall_10 is None or recall_10 > best_recall_10:
                    best_recall_10 = recall_10
                    best_config = result
    except Exception as e:
        continue

if best_config:
    print(f"   配置: {best_config['config_key']}")
    print(f"   策略: {best_config['strategy']}, 归一化: {best_config['normalization']}, 权重: {best_config['sentiment_weight']}")
    print(f"   Recall@10: {best_recall_10:.4f}")
    if best_config['metrics'].get('mrr'):
        print(f"   MRR: {best_config['metrics']['mrr']:.4f}")
else:
    print("   暂无有效结果")
PYEOF
        echo ""
    fi
else
    echo "⚠️  结果目录不存在: $RESULTS_DIR"
fi

echo ""
echo "================================================================================"

