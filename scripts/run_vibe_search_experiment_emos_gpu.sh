#!/bin/bash

LOG_FILE="test_vibe_search_experiment_emos_gpu_run.log"
SCRIPT_PATH="test/test_vibe_search_experiment_emos_gpu.py"

echo "🚀 启动 Emos模型 + GPU 实验（后台运行，PID: $$）"
echo ""
echo "📊 实验对比："
echo "  - 当前运行: LLM API 版本（PID: 47363）"
echo "  - 新启动: Emos模型 + MPS GPU 版本"
echo ""
echo "📊 实时监控命令："
echo "  tail -f ${LOG_FILE} | grep -E '(提取情绪向量|Recall@1|实验完成|✅|GPU|MPS)'"
echo ""
echo "📁 输出目录："
echo "  outputs/vibe_search_experiment_emos_gpu/"
echo ""

# 直接使用python运行脚本（使用绝对路径确保正常工作）
cd "$(dirname "${SCRIPT_PATH}")/.." && nohup python3 "${SCRIPT_PATH}" > "${LOG_FILE}" 2>&1 &
NEW_PID=$!

echo "✅ Emos+GPU 实验已启动（后台运行，PID: ${NEW_PID}）"
echo ""
echo "🔄 检查两个实验的状态："
echo "  ps aux | grep -E '(test_vibe_search_experiment|emos_gpu)' | grep -v grep"
echo ""
echo "📊 对比两个实验的进度："
echo "  echo '=== LLM API 版本 ===' && tail -3 test_vibe_search_experiment_run.log"
echo "  echo '=== Emos GPU 版本 ===' && tail -3 ${LOG_FILE}"
echo ""
