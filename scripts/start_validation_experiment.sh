#!/bin/bash
# 启动HyperAmy修复验证实验

PROJECT_ROOT="/public/jiangh/HyperAmy"
SCRIPT_PATH="${PROJECT_ROOT}/test/test_hyperamy_quick_validation.py"
LOG_FILE="${PROJECT_ROOT}/test_hyperamy_quick_validation.log"
PID_FILE="${PROJECT_ROOT}/hyperamy_validation.pid"
CONDA_ENV="PyTorch-2.4.1"

echo "================================================================================"
echo "🔍 启动 HyperAmy修复验证 - 小规模测试（10个查询）"
echo "================================================================================"
echo "脚本路径: $SCRIPT_PATH"
echo "日志文件: $LOG_FILE"
echo "PID文件: $PID_FILE"
echo "================================================================================"
echo ""

# 检查是否有旧进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null; then
        echo "⚠️  检测到旧的验证进程 (PID: $OLD_PID) 仍在运行。请手动停止或等待其完成。"
        echo "   如果需要强制停止，请运行: kill $OLD_PID"
        exit 1
    else
        echo "清理旧的PID文件: $PID_FILE"
        rm "$PID_FILE"
    fi
fi

# 检查必要文件是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 验证脚本不存在: $SCRIPT_PATH"
    echo "   请先同步文件到服务器"
    exit 1
fi

if [ ! -d "${PROJECT_ROOT}/outputs/three_methods_comparison_monte_cristo/hyperamy_db" ]; then
    echo "❌ HyperAmy存储目录不存在"
    echo "   请先运行完整实验或并行索引脚本"
    exit 1
fi

# 激活conda环境并启动Python脚本
echo "激活 Conda 环境: $CONDA_ENV"
echo "启动验证脚本: $SCRIPT_PATH"
echo "日志文件: $LOG_FILE"

cd "$PROJECT_ROOT" && \
source /opt/conda/etc/profile.d/conda.sh && \
conda activate "$CONDA_ENV" && \
nohup python -u "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &

# 保存PID
echo $! > "$PID_FILE"
NEW_PID=$!

echo "✅ HyperAmy验证脚本已在后台启动，PID: $NEW_PID"
echo "   你可以使用 'tail -f $LOG_FILE' 查看实时日志。"
echo "   使用 'ps aux | grep [t]est_hyperamy_quick_validation.py' 检查进程状态。"
echo "   使用 'kill $NEW_PID' 停止进程。"

echo "================================================================================"

