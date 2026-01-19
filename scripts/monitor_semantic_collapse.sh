#!/bin/bash
# 监控语义崩溃协议触发的实时日志

LOG_FILE="test_vibe_search_experiment_emos_gpu_run.log"

echo "=================================================================="
echo "🔍 语义崩溃协议监控 - 实时日志"
echo "=================================================================="
echo ""
echo "监控文件: $LOG_FILE"
echo ""
echo "关键指标："
echo "  1. ⚠️  Semantic Collapse Detected（语义崩溃警告）"
echo "  2. W_emo 值（应该从 0.3 提升到 > 0.8）"
echo "  3. Recall@1 最终结果"
echo ""
echo "=================================================================="
echo ""

tail -f "$LOG_FILE" 2>/dev/null | grep --line-buffered -E "(Semantic Collapse|⚠️|W_emo|Dynamic Weighting.*W_emo=|Recall@1|实验完成)" | while read line; do
    # 高亮语义崩溃警告
    if echo "$line" | grep -q "Semantic Collapse"; then
        echo "🔥 [崩溃触发] $line"
    elif echo "$line" | grep -q "W_emo=0\.[89]"; then
        echo "⚡ [高权重] $line"
    elif echo "$line" | grep -q "Recall@1"; then
        echo "📊 [结果] $line"
    else
        echo "$line"
    fi
done
