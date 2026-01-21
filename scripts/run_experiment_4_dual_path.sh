#!/bin/bash
# Experiment 4: HyperAmy V2 - Adaptive Dual-Path Retrieval
# 自适应双路检索实验

cd /public/jiangh/HyperAmy || exit 1

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate PyTorch-2.4.1

# 🔧 关键修复：防止 torch 导入死锁
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 从.env文件读取API_KEY
if [ -f .env ]; then
    export API_KEY=$(grep "^API_KEY=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    export OPENAI_API_KEY=$API_KEY
    echo "✅ API_KEY 已从 .env 文件加载"
else
    echo "⚠️  警告: .env 文件不存在，请确保 API_KEY 已设置"
fi

echo "======================================================================"
echo "启动 Experiment 4: HyperAmy V2 - Adaptive Dual-Path Retrieval"
echo "======================================================================"
echo "核心特性："
echo "  - Path A: Hybrid Re-ranking (语义置信度高时)"
echo "  - Path B: Global Emotion Search (语义崩溃时)"
echo "  - 自适应切换，绕过HippoRAG召回率低的问题"
echo "======================================================================"
echo "🔧 已应用 OMP_NUM_THREADS=1 修复（防止 torch 导入死锁）"
echo ""

# 确保输出目录存在
mkdir -p outputs/vibe_search_experiment_4_dual_path

# 检查是否已有运行中的实验
if [ -f outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.pid ]; then
    OLD_PID=$(cat outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  发现已有运行中的实验 (PID: $OLD_PID)"
        read -p "是否终止旧实验并启动新的? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill $OLD_PID 2>/dev/null
            sleep 2
        else
            echo "已取消"
            exit 1
        fi
    fi
fi

# 启动实验（使用 nohup 后台运行）
echo "🚀 启动 Experiment 4..."
nohup python3 -u test/test_vibe_search_experiment_4_dual_path.py \
    > outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.log 2>&1 &

EXPERIMENT_PID=$!

echo $EXPERIMENT_PID > outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.pid

echo "✅ 实验已启动"
echo "进程ID: $EXPERIMENT_PID"
echo "日志文件: outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.log"
echo ""
echo "监控命令:"
echo "  tail -f outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.log | grep -E 'DUAL-PATH|Switching|Recall@'"
echo "  ps -p $EXPERIMENT_PID"
echo ""

sleep 3
echo "进程状态:"
ps -p $EXPERIMENT_PID -o pid,etime,%cpu,%mem,cmd --no-headers || echo "进程未找到"

echo ""
echo "最新日志（最后10行）:"
tail -10 outputs/vibe_search_experiment_4_dual_path/experiment_4_dual_path.log 2>/dev/null || echo "日志文件尚未生成"
