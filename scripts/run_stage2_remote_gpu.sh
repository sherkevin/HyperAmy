#!/bin/bash
# 二阶段训练：远程服务器GPU运行脚本

set -e

SERVER_HOST="jiangh@10.103.92.120"
SERVER_PORT="1066"
REMOTE_DIR="/public/jiangh/stage2_training"
REMOTE_PROJECT_DIR="/public/jiangh/HyperAmy"

echo "============================================================"
echo "二阶段训练：远程服务器GPU运行"
echo "============================================================"
echo "服务器: $SERVER_HOST:$SERVER_PORT"
echo "远程目录: $REMOTE_DIR"
echo ""

# 检查服务器连接
echo "【步骤1】检查服务器连接..."
ssh -p $SERVER_PORT $SERVER_HOST "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -1" || {
    echo "❌ 无法连接到服务器或GPU不可用"
    exit 1
}
echo "✅ 服务器连接正常，GPU可用"

# 创建远程目录
echo ""
echo "【步骤2】准备远程目录..."
ssh -p $SERVER_PORT $SERVER_HOST << REMOTE_SCRIPT
mkdir -p $REMOTE_DIR
mkdir -p $REMOTE_PROJECT_DIR
cd $REMOTE_PROJECT_DIR || mkdir -p $REMOTE_PROJECT_DIR
pwd
REMOTE_SCRIPT

# 同步必要文件
echo ""
echo "【步骤3】同步文件到服务器..."
rsync -avz --progress -e "ssh -p $SERVER_PORT" \
    scripts/stage2_hard_negative_training.py \
    scripts/stage2_contrastive_train.py \
    $SERVER_HOST:$REMOTE_PROJECT_DIR/scripts/ || {
    echo "❌ 文件同步失败"
    exit 1
}

# 同步数据文件（如果不存在）
echo ""
echo "【步骤4】检查并同步数据文件..."
ssh -p $SERVER_PORT $SERVER_HOST << REMOTE_SCRIPT
if [ ! -f "$REMOTE_PROJECT_DIR/data/benchmarks/instinct_qa.json" ]; then
    echo "需要同步数据文件..."
    exit 1
fi
REMOTE_SCRIPT

if [ $? -ne 0 ]; then
    echo "同步数据文件..."
    rsync -avz --progress -e "ssh -p $SERVER_PORT" \
        data/benchmarks/instinct_qa.json \
        data/processed/got_amygdala.jsonl \
        data/training/entity_granularity/entity_granularity_v2_full.jsonl \
        $SERVER_HOST:$REMOTE_PROJECT_DIR/data/ 2>/dev/null || {
        echo "⚠️  数据文件同步失败，将在服务器上检查"
    }
fi

# 检查模型文件
echo ""
echo "【步骤5】检查模型文件..."
MODEL_PATH="~/Desktop/best_model.pt"
ssh -p $SERVER_PORT $SERVER_HOST "test -f $MODEL_PATH || test -f /public/jiangh/emos/checkpoints/best_model.pt" || {
    echo "⚠️  模型文件可能不存在，将尝试使用默认路径"
}

# 在服务器上运行实验
echo ""
echo "============================================================"
echo "【步骤6】在服务器上运行实验（GPU）"
echo "============================================================"
echo "开始时间: $(date)"
echo ""

ssh -p $SERVER_PORT $SERVER_HOST << REMOTE_SCRIPT
set -e
cd $REMOTE_PROJECT_DIR

# 激活conda环境（如果需要）
if command -v conda &> /dev/null; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate PyTorch-2.4.1 2>/dev/null || true
fi

# 设置Python路径
export PYTHONPATH="\$PYTHONPATH:$REMOTE_PROJECT_DIR:$REMOTE_PROJECT_DIR/emos-master"

# 创建输出目录
mkdir -p $REMOTE_DIR

echo "开始运行数据构造和评估..."
python3 scripts/stage2_hard_negative_training.py \
    --qa_file data/benchmarks/instinct_qa.json \
    --chunks_file data/processed/got_amygdala.jsonl \
    --entity_file data/training/entity_granularity/entity_granularity_v2_full.jsonl \
    --model_checkpoint ~/Desktop/best_model.pt \
    --output_dir $REMOTE_DIR \
    --device cuda \
    --num_negative_contexts 5 \
    --hard_negative_threshold 3 \
    2>&1 | tee $REMOTE_DIR/experiment.log

echo ""
echo "检查难例数量..."
if [ -f "$REMOTE_DIR/hard_negatives.jsonl" ]; then
    NUM_HARD=\$(wc -l < $REMOTE_DIR/hard_negatives.jsonl)
    echo "筛选出的难例数量: \$NUM_HARD"
    
    if [ \$NUM_HARD -gt 0 ]; then
        echo ""
        echo "开始对比学习训练..."
        python3 scripts/stage2_contrastive_train.py \
            --hard_negatives_file $REMOTE_DIR/hard_negatives.jsonl \
            --model_checkpoint ~/Desktop/best_model.pt \
            --output_dir $REMOTE_DIR \
            --device cuda \
            --batch_size 8 \
            --epochs 5 \
            --learning_rate 1e-5 \
            --margin 0.1 \
            2>&1 | tee -a $REMOTE_DIR/experiment.log
    else
        echo "⚠️  没有难例，跳过训练步骤"
    fi
else
    echo "⚠️  难例文件不存在"
fi

echo ""
echo "实验完成时间: \$(date)"
REMOTE_SCRIPT

# 同步结果回本地
echo ""
echo "============================================================"
echo "【步骤7】同步结果回本地"
echo "============================================================"

LOCAL_OUTPUT_DIR="outputs/stage2_training_remote"
mkdir -p "$LOCAL_OUTPUT_DIR"

rsync -avz --progress -e "ssh -p $SERVER_PORT" \
    $SERVER_HOST:$REMOTE_DIR/ \
    $LOCAL_OUTPUT_DIR/ || {
    echo "⚠️  结果同步失败，但实验可能已完成"
}

echo ""
echo "============================================================"
echo "✅ 实验完成！"
echo "============================================================"
echo "结果文件位置:"
echo "  远程: $REMOTE_DIR"
echo "  本地: $LOCAL_OUTPUT_DIR"
echo ""
echo "查看日志:"
echo "  cat $LOCAL_OUTPUT_DIR/experiment.log"
echo ""

