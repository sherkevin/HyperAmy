#!/bin/bash
# HyperAmy并行运行脚本 - 可以与主实验并行执行

cd /public/jiangh/HyperAmy || exit 1

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate PyTorch-2.4.1

# 检查是否已经有HyperAmy存储（避免重复运行）
if [ -d "outputs/three_methods_comparison_monte_cristo/hyperamy_db" ]; then
    echo "⚠️  HyperAmy存储已存在，是否继续运行？"
    echo "   如果继续，将覆盖现有存储"
    read -p "继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

# 使用nohup后台运行
echo "🚀 启动HyperAmy并行索引（GPU加速）..."
nohup python -u test/test_hyperamy_parallel.py > test_hyperamy_parallel.log 2>&1 &
HYPERAMY_PID=$!

echo "HyperAmy进程已启动，PID: $HYPERAMY_PID"
echo "日志文件: test_hyperamy_parallel.log"
echo ""
echo "监控命令:"
echo "  tail -f test_hyperamy_parallel.log"
echo "  ps -p $HYPERAMY_PID"

# 保存PID
echo $HYPERAMY_PID > test_hyperamy_parallel.pid

sleep 3
echo ""
echo "进程状态:"
ps -p $HYPERAMY_PID -o pid,etime,%cpu,%mem,cmd --no-headers || echo "进程未找到"

echo ""
echo "最新日志（最后10行）:"
tail -10 test_hyperamy_parallel.log 2>/dev/null || echo "日志文件尚未生成"

