#!/bin/bash
# 运行 Experiment 3: Final Fusion (LLM API + 语义崩溃协议)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🚀 启动 Experiment 3: Final Fusion (LLM API + 语义崩溃协议)"
echo ""
echo "核心特性："
echo "  ✅ 语义崩溃协议已激活"
echo "  ✅ LLM API提取情绪向量（高I_q值）"
echo "  ✅ 自动复用缓存（.cache/emotion_vectors）"
echo ""

# 检查API Key
if [ -z "$API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告: 未检测到API_KEY环境变量"
    echo "   如果使用LLM API，请确保设置了API_KEY或OPENAI_API_KEY"
    echo ""
fi

# 检查缓存目录
CACHE_DIR=".cache/emotion_vectors"
if [ -d "$CACHE_DIR" ]; then
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    echo "✅ 检测到情绪向量缓存: $CACHE_DIR ($CACHE_SIZE)"
    echo "   将自动复用缓存，节省API费用"
    echo ""
else
    echo "ℹ️  缓存目录不存在，将创建新缓存"
    echo "   首次运行将调用API提取情绪向量"
    echo ""
fi

# 启动实验（后台运行）
LOG_FILE="outputs/vibe_search_experiment_3_final/experiment_3_final.log"
nohup python3 test/test_vibe_search_experiment_3_final.py > "$LOG_FILE" 2>&1 &
EXPERIMENT_PID=$!

echo "✅ Experiment 3 已启动（后台运行，PID: $EXPERIMENT_PID）"
echo ""
echo "📊 实时监控命令："
echo "  tail -f $LOG_FILE | grep -E '(Semantic Collapse|COLLAPSE PROTOCOL|W_emo|Recall@1|I_q|实验完成)'"
echo ""
echo "📁 输出目录："
echo "  outputs/vibe_search_experiment_3_final/"
echo ""
echo "🔄 检查实验状态："
echo "  ps aux | grep test_vibe_search_experiment_3_final | grep -v grep"
echo ""
echo "📊 查看关键日志："
echo "  tail -f $LOG_FILE | grep -E '(Query.*I_q|COLLAPSE|Recall@1)'"
