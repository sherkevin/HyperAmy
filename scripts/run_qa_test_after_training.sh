#!/bin/bash
# 训练完成后运行QA测试对比

SERVER="jiangh@10.103.92.120"
PORT="1066"
REMOTE_PROJECT_DIR="/public/jiangh/HyperAmy"
REMOTE_EMOS_DIR="/public/jiangh/emos"

echo "============================================================"
echo "训练完成后运行QA测试对比"
echo "============================================================"

# 同步代码到服务器
echo "【步骤1】同步测试代码到服务器..."
rsync -avz --progress -e "ssh -p $PORT" \
    test/test_three_methods_comparison_monte_cristo.py \
    particle/emotion_v3.py \
    particle/emos_wrapper.py \
    $SERVER:$REMOTE_PROJECT_DIR/ 2>&1 | tail -5

# 在服务器上运行测试
ssh -p $PORT $SERVER << 'REMOTE_SCRIPT'
cd /public/jiangh/HyperAmy

echo ""
echo "【步骤2】检查训练完成的模型..."
MODEL_CHECKPOINT="/public/jiangh/emos/checkpoints/best_model.pt"
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "❌ 模型checkpoint不存在: $MODEL_CHECKPOINT"
    echo "   请先完成训练"
    exit 1
fi

echo "✅ 找到模型: $MODEL_CHECKPOINT"
ls -lh "$MODEL_CHECKPOINT"

echo ""
echo "【步骤3】运行QA测试对比..."
echo "   对比四种方法："
echo "   1. HyperAmy (LLM抽取情绪向量)"
echo "   2. HyperAmy-Emos (7B模型抽取情绪向量)"
echo "   3. HippoRAG (纯语义检索)"
echo "   4. Fusion (语义+情绪混合检索)"

# 设置环境变量
export EMOS_PATH="/public/jiangh/emos"
export PYTHONPATH="/public/jiangh/HyperAmy:/public/jiangh/emos:$PYTHONPATH"

# 运行测试（后台运行）
LOG_FILE="test_four_methods_comparison_7b_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"

nohup python3 test/test_three_methods_comparison_monte_cristo.py > "$LOG_FILE" 2>&1 &
TEST_PID=$!

echo "✅ 测试已启动（PID: $TEST_PID）"
echo ""
echo "监控测试进度："
echo "  tail -f $LOG_FILE"

REMOTE_SCRIPT

echo ""
echo "✅ 测试已启动！"
echo ""
echo "查看进度："
echo "  ssh $SERVER -p $PORT"
echo "  cd $REMOTE_PROJECT_DIR"
echo "  tail -f test_four_methods_comparison_7b_*.log"
