#!/bin/bash
# 实时监控两个实验的进度

echo "=================================================================="
echo "实时监控：实验1 (LLM API) vs 实验2 (Emos GPU)"
echo "=================================================================="
echo ""

EXP1_LOG="test_vibe_search_experiment_run.log"
EXP2_LOG="test_vibe_search_experiment_emos_gpu_run.log"

# 检查进程状态
echo "【进程状态】"
EXP1_PID=$(ps aux | grep "test_vibe_search_experiment_final" | grep -v grep | awk '{print $2}')
EXP2_PID=$(ps aux | grep "test_vibe_search_experiment_emos_gpu" | grep -v grep | awk '{print $2}')

if [ -n "$EXP1_PID" ]; then
    EXP1_CPU=$(ps -p $EXP1_PID -o %cpu= | tr -d ' ')
    EXP1_MEM=$(ps -p $EXP1_PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')
    echo "  实验1 (LLM API): ✅ 运行中 (PID: $EXP1_PID, CPU: ${EXP1_CPU}%, MEM: $EXP1_MEM)"
else
    echo "  实验1 (LLM API): ❌ 未运行"
fi

if [ -n "$EXP2_PID" ]; then
    EXP2_CPU=$(ps -p $EXP2_PID -o %cpu= | tr -d ' ')
    EXP2_MEM=$(ps -p $EXP2_PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')
    echo "  实验2 (Emos GPU): ✅ 运行中 (PID: $EXP2_PID, CPU: ${EXP2_CPU}%, MEM: $EXP2_MEM)"
else
    echo "  实验2 (Emos GPU): ❌ 未运行"
fi

echo ""

# 检查进度
echo "【当前进度】"

# 实验1进度
if [ -f "$EXP1_LOG" ]; then
    EXP1_PROGRESS=$(tail -50 "$EXP1_LOG" | grep -oE "HyperAmy-Hybrid检索.*[0-9]+/[0-9]+" | tail -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
    if [ -n "$EXP1_PROGRESS" ]; then
        EXP1_CURRENT=$(echo $EXP1_PROGRESS | cut -d'/' -f1)
        EXP1_TOTAL=$(echo $EXP1_PROGRESS | cut -d'/' -f2)
        EXP1_PCT=$(echo "scale=1; $EXP1_CURRENT * 100 / $EXP1_TOTAL" | bc)
        echo "  实验1: $EXP1_PROGRESS ($EXP1_PCT%)"
    else
        EXP1_STAGE=$(tail -10 "$EXP1_LOG" | grep -E "(步骤|初始化|索引|检索)" | tail -1 | sed 's/^[[:space:]]*//' | cut -c1-50)
        echo "  实验1: $EXP1_STAGE"
    fi
else
    echo "  实验1: 日志文件不存在"
fi

# 实验2进度
if [ -f "$EXP2_LOG" ]; then
    EXP2_PROGRESS=$(tail -50 "$EXP2_LOG" | grep -oE "HyperAmy-Hybrid检索.*[0-9]+/[0-9]+" | tail -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
    if [ -n "$EXP2_PROGRESS" ]; then
        EXP2_CURRENT=$(echo $EXP2_PROGRESS | cut -d'/' -f1)
        EXP2_TOTAL=$(echo $EXP2_PROGRESS | cut -d'/' -f2)
        EXP2_PCT=$(echo "scale=1; $EXP2_CURRENT * 100 / $EXP2_TOTAL" | bc)
        echo "  实验2: $EXP2_PROGRESS ($EXP2_PCT%)"
    else
        EXP2_STAGE=$(tail -10 "$EXP2_LOG" | grep -E "(步骤|初始化|索引|检索)" | tail -1 | sed 's/^[[:space:]]*//' | cut -c1-50)
        echo "  实验2: $EXP2_STAGE"
    fi
else
    echo "  实验2: 日志文件不存在"
fi

echo ""

# 检查结果文件
echo "【结果文件】"
if [ -f "outputs/vibe_search_experiment/results.json" ]; then
    echo "  ✅ 实验1结果已生成"
    python3 -c "import json; d=json.load(open('outputs/vibe_search_experiment/results.json')); print(f\"    HippoRAG Recall@1: {d.get('hipporag',{}).get('recall_at_1',0):.2%}\"); print(f\"    Hybrid Recall@1: {d.get('hybrid',{}).get('recall_at_1',0):.2%}\")" 2>/dev/null
else
    echo "  ⏳ 实验1结果未生成（实验进行中）"
fi

if [ -f "outputs/vibe_search_experiment_emos_gpu/results.json" ]; then
    echo "  ✅ 实验2结果已生成"
    python3 -c "import json; d=json.load(open('outputs/vibe_search_experiment_emos_gpu/results.json')); print(f\"    HippoRAG Recall@1: {d.get('hipporag',{}).get('recall_at_1',0):.2%}\"); print(f\"    Hybrid Recall@1: {d.get('hybrid',{}).get('recall_at_1',0):.2%}\")" 2>/dev/null
else
    echo "  ⏳ 实验2结果未生成（实验进行中）"
fi

echo ""
echo "=================================================================="
echo "使用 './scripts/compare_both_experiments.sh' 查看完整对比"
echo "=================================================================="
