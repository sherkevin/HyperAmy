#!/bin/bash
# 对比两个实验的结果

echo "=================================================================="
echo "实验对比：LLM API版本 vs Emos GPU版本"
echo "=================================================================="
echo ""

EXP1_RESULTS="outputs/vibe_search_experiment/results.json"
EXP2_RESULTS="outputs/vibe_search_experiment_emos_gpu/results.json"

# 检查实验1结果
if [ -f "$EXP1_RESULTS" ]; then
    echo "✅ 实验1结果文件存在"
    python3 << 'PYEOF'
import json
import sys

try:
    exp1 = json.load(open('outputs/vibe_search_experiment/results.json'))
    print("\n【实验1: LLM API版本】")
    print(f"  HippoRAG Recall@1: {exp1.get('hipporag', {}).get('recall_at_1', 0):.2%}")
    print(f"  HyperAmy Recall@1: {exp1.get('hyperamy', {}).get('recall_at_1', 0):.2%}")
    print(f"  Hybrid Recall@1: {exp1.get('hybrid', {}).get('recall_at_1', 0):.2%}")
except Exception as e:
    print(f"❌ 读取实验1结果失败: {e}")
PYEOF
else
    echo "❌ 实验1结果文件不存在（实验可能还在进行中）"
fi

echo ""

# 检查实验2结果
if [ -f "$EXP2_RESULTS" ]; then
    echo "✅ 实验2结果文件存在"
    python3 << 'PYEOF'
import json
import sys

try:
    exp2 = json.load(open('outputs/vibe_search_experiment_emos_gpu/results.json'))
    print("\n【实验2: Emos GPU版本】")
    print(f"  HippoRAG Recall@1: {exp2.get('hipporag', {}).get('recall_at_1', 0):.2%}")
    print(f"  HyperAmy Recall@1: {exp2.get('hyperamy', {}).get('recall_at_1', 0):.2%}")
    print(f"  Hybrid Recall@1: {exp2.get('hybrid', {}).get('recall_at_1', 0):.2%}")
except Exception as e:
    print(f"❌ 读取实验2结果失败: {e}")
PYEOF
else
    echo "❌ 实验2结果文件不存在（实验可能还在进行中）"
fi

echo ""

# 如果两个结果都存在，生成对比表格
if [ -f "$EXP1_RESULTS" ] && [ -f "$EXP2_RESULTS" ]; then
    echo "=================================================================="
    echo "对比表格"
    echo "=================================================================="
    python3 << 'PYEOF'
import json

exp1 = json.load(open('outputs/vibe_search_experiment/results.json'))
exp2 = json.load(open('outputs/vibe_search_experiment_emos_gpu/results.json'))

print(f"{'方法':<20} {'实验1 (LLM API)':<20} {'实验2 (Emos GPU)':<20} {'差异':<15}")
print("-" * 75)

methods = [
    ('hipporag', 'HippoRAG'),
    ('hyperamy', 'HyperAmy'),
    ('hybrid', 'Hybrid')
]

for method_key, method_name in methods:
    r1 = exp1.get(method_key, {}).get('recall_at_1', 0)
    r2 = exp2.get(method_key, {}).get('recall_at_1', 0)
    diff = r2 - r1
    diff_str = f"{diff:+.2%}" if diff != 0 else "0.00%"
    print(f"{method_name:<20} {r1:>18.2%} {r2:>18.2%} {diff_str:>14}")

print("=" * 75)
PYEOF
fi

echo ""
echo "📊 实时监控命令："
echo "  tail -f test_vibe_search_experiment_run.log | grep -E '(Recall@1|实验完成)'"
echo "  tail -f test_vibe_search_experiment_emos_gpu_run.log | grep -E '(Recall@1|实验完成)'"
echo ""
echo "🔄 重新运行对比："
echo "  ./scripts/compare_both_experiments.sh"
