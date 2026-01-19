#!/bin/bash
# Vibe Search数据集生成脚本启动器

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 设置日志文件
LOG_FILE="$PROJECT_ROOT/generate_vibe_dataset.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Vibe Search数据集生成"
echo "============================================================"
echo "项目根目录: $PROJECT_ROOT"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python3"
    exit 1
fi

# 检查数据文件
CHUNKS_FILE="$PROJECT_ROOT/data/training/monte_cristo_train_full.jsonl"
if [ ! -f "$CHUNKS_FILE" ]; then
    echo "❌ Chunks文件不存在: $CHUNKS_FILE"
    exit 1
fi

echo "✅ Chunks文件存在: $CHUNKS_FILE"

# 检查环境变量
if [ -z "$API_KEY" ]; then
    echo "⚠️  API_KEY环境变量未设置，将使用.env文件中的配置"
fi

# 运行生成脚本（后台运行）
echo ""
echo "🚀 启动数据集生成脚本..."
echo ""

python3 scripts/generate_vibe_dataset.py \
    --chunks_file "$CHUNKS_FILE" \
    --output_file "$PROJECT_ROOT/data/public_benchmark/monte_cristo_vibe_search.json" \
    --max_queries 50 \
    --emotion_threshold 8.0 \
    --max_workers 5 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 数据集生成完成！"
    echo "输出文件: $PROJECT_ROOT/data/public_benchmark/monte_cristo_vibe_search.json"
else
    echo "❌ 数据集生成失败（退出码: $EXIT_CODE）"
    echo "请查看日志: $LOG_FILE"
fi
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE
