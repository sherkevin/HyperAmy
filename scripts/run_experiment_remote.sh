#!/bin/bash
# 在远程服务器上运行严谨实验（使用GPU）

REMOTE_HOST="jiangh@10.103.16.22"
REMOTE_PATH="/media/data4/jiangh/Amygdala/hyperamy_source"

echo "=========================================="
echo "在远程服务器上启动严谨实验"
echo "=========================================="

# 检查GPU状态
echo "【1】检查GPU状态..."
ssh "$REMOTE_HOST" "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -7"

# 启动实验（使用screen保持会话）
echo ""
echo "【2】启动实验..."
ssh "$REMOTE_HOST" "cd $REMOTE_PATH && screen -dmS experiment python3 -m test.test_rigorous_comparison_save"

# 等待几秒确认启动
sleep 3

# 检查实验是否在运行
echo ""
echo "【3】检查实验状态..."
ssh "$REMOTE_HOST" "cd $REMOTE_PATH && screen -ls | grep experiment && echo '---' && ps aux | grep test_rigorous_comparison_save | grep -v grep"

# 检查日志
echo ""
echo "【4】查看最新日志（最后10行）..."
ssh "$REMOTE_HOST" "cd $REMOTE_PATH && LATEST_LOG=\$(ls -t outputs/rigorous_experiment/logs/*.log 2>/dev/null | head -1) && if [ -n \"\$LATEST_LOG\" ]; then tail -10 \"\$LATEST_LOG\"; else echo '暂无日志文件'; fi"

echo ""
echo "=========================================="
echo "✅ 实验已启动"
echo "=========================================="
echo ""
echo "监控命令:"
echo "  ssh $REMOTE_HOST 'cd $REMOTE_PATH && screen -r experiment'"
echo "  或查看日志: tail -f outputs/rigorous_experiment/logs/experiment_*.log"

