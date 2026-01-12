#!/bin/bash
# 运行HyperAmy修复验证脚本（小规模测试）

PROJECT_ROOT="/public/jiangh/HyperAmy"
SCRIPT_PATH="${PROJECT_ROOT}/test/test_hyperamy_quick_validation.py"
LOG_FILE="${PROJECT_ROOT}/test_hyperamy_quick_validation.log"
CONDA_ENV="PyTorch-2.4.1"

echo "================================================================================"
echo "🔍 HyperAmy修复验证 - 小规模测试（10个查询）"
echo "================================================================================"
echo "脚本路径: $SCRIPT_PATH"
echo "日志文件: $LOG_FILE"
echo "================================================================================"
echo ""

cd "$PROJECT_ROOT" && \
source /opt/conda/etc/profile.d/conda.sh && \
conda activate "$CONDA_ENV" && \
python -u "$SCRIPT_PATH" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 验证完成！请查看日志和结果文件"
    echo "   日志文件: $LOG_FILE"
    echo "   结果文件: outputs/three_methods_comparison_monte_cristo/hyperamy_validation_results.json"
else
    echo "❌ 验证失败，退出码: $EXIT_CODE"
    echo "   请查看日志: $LOG_FILE"
fi
echo "================================================================================"

exit $EXIT_CODE

