#!/bin/bash
# 二阶段训练：大规模实验启动脚本

set -e

echo "============================================================"
echo "二阶段训练：大规模实验"
echo "============================================================"

# 配置参数
QA_FILE="data/benchmarks/instinct_qa.json"
CHUNKS_FILE="data/processed/got_amygdala.jsonl"
ENTITY_FILE="data/training/entity_granularity/entity_granularity_v2_full.jsonl"
MODEL_CHECKPOINT="$HOME/Desktop/best_model.pt"
OUTPUT_DIR="outputs/stage2_training"
DEVICE="${1:-cpu}"  # 第一个参数，默认为cpu
NUM_NEGATIVE="${2:-5}"  # 第二个参数，默认为5

echo ""
echo "配置:"
echo "  QA文件: $QA_FILE"
echo "  Chunks文件: $CHUNKS_FILE"
echo "  实体文件: $ENTITY_FILE"
echo "  模型: $MODEL_CHECKPOINT"
echo "  输出目录: $OUTPUT_DIR"
echo "  设备: $DEVICE"
echo "  负样本数: $NUM_NEGATIVE"
echo ""

# 检查文件
if [ ! -f "$QA_FILE" ]; then
    echo "❌ QA文件不存在: $QA_FILE"
    exit 1
fi

if [ ! -f "$CHUNKS_FILE" ]; then
    echo "❌ Chunks文件不存在: $CHUNKS_FILE"
    exit 1
fi

if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "❌ 模型文件不存在: $MODEL_CHECKPOINT"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "步骤1: 数据构造和评估（生成难例）"
echo "============================================================"

python3 scripts/stage2_hard_negative_training.py \
    --qa_file "$QA_FILE" \
    --chunks_file "$CHUNKS_FILE" \
    --entity_file "$ENTITY_FILE" \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --num_negative_contexts "$NUM_NEGATIVE" \
    --hard_negative_threshold 3

if [ $? -ne 0 ]; then
    echo "❌ 步骤1失败"
    exit 1
fi

echo ""
echo "============================================================"
echo "步骤2: 检查难例数量"
echo "============================================================"

HARD_NEGATIVES_FILE="$OUTPUT_DIR/hard_negatives.jsonl"
if [ -f "$HARD_NEGATIVES_FILE" ]; then
    NUM_HARD=$(wc -l < "$HARD_NEGATIVES_FILE")
    echo "筛选出的难例数量: $NUM_HARD"
    
    if [ $NUM_HARD -eq 0 ]; then
        echo "⚠️  没有难例，跳过训练步骤"
        exit 0
    fi
else
    echo "❌ 难例文件不存在: $HARD_NEGATIVES_FILE"
    exit 1
fi

echo ""
echo "============================================================"
echo "步骤3: 对比学习训练"
echo "============================================================"

python3 scripts/stage2_contrastive_train.py \
    --hard_negatives_file "$HARD_NEGATIVES_FILE" \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch_size 4 \
    --epochs 5 \
    --learning_rate 1e-5 \
    --margin 0.1

if [ $? -ne 0 ]; then
    echo "❌ 步骤3失败"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ 大规模实验完成！"
echo "============================================================"
echo "结果文件:"
echo "  - 构造数据: $OUTPUT_DIR/constructed_data.jsonl"
echo "  - 评估结果: $OUTPUT_DIR/evaluation_results.json"
echo "  - 难例数据: $OUTPUT_DIR/hard_negatives.jsonl"
echo "  - 模型checkpoints: $OUTPUT_DIR/checkpoints/"
echo ""

