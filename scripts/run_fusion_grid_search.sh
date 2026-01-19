#!/bin/bash
# 启动Fusion策略网格搜索实验（后台运行）

# 定义项目根目录
PROJECT_ROOT="/public/jiangh/HyperAmy"
LOG_FILE="${PROJECT_ROOT}/fusion_strategy_grid_search.log"
PID_FILE="${PROJECT_ROOT}/fusion_grid_search.pid"
SCRIPT_PATH="${PROJECT_ROOT}/test/test_fusion_strategy_grid_search.py"
CONDA_ENV="PyTorch-2.4.1"

echo "================================================================================"
echo "🚀 启动 Fusion 策略网格搜索实验（后台运行）"
echo "================================================================================"
echo "项目目录: $PROJECT_ROOT"
echo "脚本路径: $SCRIPT_PATH"
echo "日志文件: $LOG_FILE"
echo "PID文件: $PID_FILE"
echo "Conda环境: $CONDA_ENV"
echo "================================================================================"
echo ""

# 检查是否有旧进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  检测到旧的网格搜索进程 (PID: $OLD_PID) 仍在运行。"
        echo "   如果需要强制停止，请运行: kill $OLD_PID"
        echo "   或者等待其完成。"
        read -p "是否继续启动新进程? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        echo "   停止旧进程..."
        kill $OLD_PID 2>/dev/null
        sleep 2
    else
        echo "清理旧的PID文件: $PID_FILE"
        rm "$PID_FILE"
    fi
fi

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 脚本文件不存在: $SCRIPT_PATH"
    exit 1
fi

# 激活conda环境并启动Python脚本
echo "激活 Conda 环境: $CONDA_ENV"
echo "启动 Python 脚本: $SCRIPT_PATH"
echo ""

cd "$PROJECT_ROOT" && \
source /opt/conda/etc/profile.d/conda.sh && \
conda activate "$CONDA_ENV" && \
nohup python -u "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &

# 保存PID
NEW_PID=$!
echo $NEW_PID > "$PID_FILE"

echo "✅ Fusion 策略网格搜索实验已在后台启动"
echo "   PID: $NEW_PID"
echo "   日志文件: $LOG_FILE"
echo ""
echo "📊 监控命令:"
echo "   查看实时日志: tail -f $LOG_FILE"
echo "   查看进程状态: ps aux | grep [t]est_fusion_strategy_grid_search.py"
echo "   停止实验: kill $NEW_PID"
echo ""
echo "📁 结果文件位置:"
echo "   结果目录: $PROJECT_ROOT/outputs/fusion_strategy_grid_search/"
echo "   进度文件: $PROJECT_ROOT/outputs/fusion_strategy_grid_search/progress.json"
echo "   汇总报告: $PROJECT_ROOT/outputs/fusion_strategy_grid_search/grid_search_summary.json"
echo ""
echo "================================================================================"

