#!/bin/bash
# 在远程服务器上运行实验（GPU加速，极致速度）

REMOTE_HOST="jiangh@10.103.16.22"
REMOTE_PATH="/media/data4/jiangh/Amygdala/hyperamy_source"
PYTHON_BIN="/media/data4/jiangh/conda/envs/Amygdala/bin/python"

echo "======================================================================"
echo "启动云端GPU加速实验（极致速度）"
echo "======================================================================"

# 检查并启动
ssh "$REMOTE_HOST" << EOF
cd $REMOTE_PATH
source /media/data4/jiangh/conda/etc/profile.d/conda.sh
conda activate Amygdala

# 设置GPU环境变量（使用所有GPU）
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 设置API密钥（从配置文件读取，需要先设置PYTHONPATH）
export PYTHONPATH="$REMOTE_PATH:$PYTHONPATH"
export OPENAI_API_KEY=\$($PYTHON_BIN -c "import sys; sys.path.insert(0, '$REMOTE_PATH'); from llm.config import API_KEY; print(API_KEY)" 2>/dev/null || echo "")
export API_KEY=\$OPENAI_API_KEY

# 停止旧实验
if [ -f test_monte_cristo_comparison_remote.pid ]; then
    OLD_PID=\$(cat test_monte_cristo_comparison_remote.pid)
    if ps -p \$OLD_PID > /dev/null 2>&1; then
        echo "停止旧实验..."
        kill \$OLD_PID 2>/dev/null
        sleep 2
    fi
fi

# 启动实验（使用nohup后台运行，设置PYTHONPATH）
echo "启动实验..."
cd $REMOTE_PATH
export PYTHONPATH="$REMOTE_PATH:$PYTHONPATH"
nohup $PYTHON_BIN -u test/test_monte_cristo_comparison.py \
    --qa-file data/public_benchmark/monte_cristo_qa_full.json \
    --chunks-file data/training/monte_cristo_train_full.jsonl \
    --output results/monte_cristo_comparison_remote.json \
    --k 5 \
    > test_monte_cristo_comparison_remote.log 2>&1 &

echo \$! > test_monte_cristo_comparison_remote.pid
echo "✅ 实验已启动 (PID: \$(cat test_monte_cristo_comparison_remote.pid))"
EOF

sleep 3

# 检查状态
echo ""
echo "【实验状态】"
ssh "$REMOTE_HOST" << EOF
cd $REMOTE_PATH
if [ -f test_monte_cristo_comparison_remote.pid ]; then
    PID=\$(cat test_monte_cristo_comparison_remote.pid)
    if ps -p \$PID > /dev/null 2>&1; then
        echo "✅ 运行中 (PID: \$PID)"
        ps -p \$PID -o pid,etime,pcpu,pmem,command | grep -v PID
    else
        echo "❌ 启动失败"
    fi
fi
EOF

echo ""
echo "【GPU状态】"
ssh "$REMOTE_HOST" "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader" | head -3

echo ""
echo "【最新日志】"
ssh "$REMOTE_HOST" "cd $REMOTE_PATH && tail -10 test_monte_cristo_comparison_remote.log 2>/dev/null"

echo ""
echo "======================================================================"
echo "监控命令:"
echo "  ssh $REMOTE_HOST 'tail -f $REMOTE_PATH/test_monte_cristo_comparison_remote.log'"
echo "  ./scripts/monitor_both_experiments.sh"
echo "======================================================================"

