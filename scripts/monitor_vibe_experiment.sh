#!/bin/bash
# Vibe Search实验实时监控脚本

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/test_vibe_search_experiment.log"
LIVE_LOG="$PROJECT_ROOT/test_vibe_search_experiment_live.log"

echo "============================================================"
echo "Vibe Search实验实时监控"
echo "============================================================"

# 检查进程是否运行
PID=$(ps aux | grep "test_vibe_search_experiment.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    echo "✅ 实验运行中（PID: $PID）"
else
    echo "⚠️  实验进程未找到（可能已完成）"
fi

echo ""
echo "📊 关键指标监控："
echo "------------------------------------------------------------"

# 检查权重翻转现象（Dynamic Weighting日志）
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "【权重翻转现象 - The Weight Flip】"
    echo "最近的Dynamic Weighting日志："
    tail -200 "$LOG_FILE" 2>/dev/null | grep "Dynamic Weighting" | tail -5 | while read line; do
        echo "  $line"
    done
    
    # 提取W_emo统计
    echo ""
    echo "【W_emo统计】"
    w_emo_values=$(tail -200 "$LOG_FILE" 2>/dev/null | grep "W_emo=" | sed -E 's/.*W_emo=([0-9.]+).*/\1/' | tail -10)
    if [ -n "$w_emo_values" ]; then
        avg=$(echo "$w_emo_values" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
        max=$(echo "$w_emo_values" | sort -n | tail -1)
        min=$(echo "$w_emo_values" | sort -n | head -1)
        echo "  最近10个查询的平均W_emo: $avg"
        echo "  最大W_emo: $max"
        echo "  最小W_emo: $min"
        
        # 统计高权重查询
        high_count=$(echo "$w_emo_values" | awk '$1 > 0.3 {count++} END {print count+0}')
        echo "  高权重查询数 (W_emo > 0.3): $high_count/10"
    fi
fi

# 检查进度
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "【实验进度】"
    total_chunks=$(grep -c "处理chunk:" "$LOG_FILE" 2>/dev/null || echo "0")
    hipporag_done=$(grep -c "HippoRAG 检索完成" "$LOG_FILE" 2>/dev/null || echo "0")
    hyperamy_done=$(grep -c "HyperAmy 检索完成" "$LOG_FILE" 2>/dev/null || echo "0")
    hybrid_done=$(grep -c "Hybrid 检索完成" "$LOG_FILE" 2>/dev/null || echo "0")
    
    echo "  已处理chunks: $total_chunks"
    echo "  HippoRAG: $([ $hipporag_done -gt 0 ] && echo "✅ 完成" || echo "⏳ 进行中")"
    echo "  HyperAmy: $([ $hyperamy_done -gt 0 ] && echo "✅ 完成" || echo "⏳ 进行中")"
    echo "  Hybrid: $([ $hybrid_done -gt 0 ] && echo "✅ 完成" || echo "⏳ 进行中")"
fi

# 检查是否有最终结果
if [ -f "$LOG_FILE" ]; then
    if grep -q "检索命中率对比" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "============================================================"
        echo "🎉 实验已完成！最终结果："
        echo "============================================================"
        grep -A 20 "检索命中率对比" "$LOG_FILE" 2>/dev/null | tail -25
    fi
fi

echo ""
echo "============================================================"
echo "实时监控命令："
echo "  tail -f $LOG_FILE | grep 'Dynamic Weighting'"
echo "  tail -f $LOG_FILE | grep -E '(W_emo|W_sem|Iq|S_sem)'"
echo "============================================================"
