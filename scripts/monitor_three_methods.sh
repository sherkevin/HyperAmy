#!/bin/bash
# 监控三种方法对比实验的实时进度

SERVER="hyperamy-server"
INTERVAL=5  # 更新间隔（秒）

echo "=================================================================================="
echo "🔍 三种方法对比实验实时监控（按 Ctrl+C 退出）"
echo "=================================================================================="
echo "更新间隔: ${INTERVAL}秒"
echo "服务器: ${SERVER}"
echo "=================================================================================="
echo ""

while true; do
    clear
    echo "=================================================================================="
    echo "🔍 三种方法对比实验实时监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================================="
    echo ""
    
    # 检查进程
    PROCESS=$(ssh ${SERVER} "ps aux | grep '[t]est_three_methods_comparison_monte_cristo' | head -1" 2>/dev/null)
    
    if [ -z "$PROCESS" ]; then
        echo "⚠️  实验进程未运行"
        
        # 检查结果文件
        RESULT_EXISTS=$(ssh ${SERVER} "test -f /public/jiangh/HyperAmy/outputs/three_methods_comparison_monte_cristo/comparison_results.json && echo 'yes' || echo 'no'" 2>/dev/null)
        
        if [ "$RESULT_EXISTS" = "yes" ]; then
            echo "✅ 实验结果文件已生成"
            echo ""
            echo "【实验结果统计】"
            ssh ${SERVER} "cd /public/jiangh/HyperAmy && python3 << 'PYEOF'
import json
from pathlib import Path

result_file = Path('outputs/three_methods_comparison_monte_cristo/comparison_results.json')
if result_file.exists():
    with open(result_file) as f:
        results = json.load(f)
    
    total = len(results)
    hipporag_hits = sum(1 for r in results if r.get('hipporag', {}).get('hit', False))
    fusion_hits = sum(1 for r in results if r.get('fusion', {}).get('hit', False))
    hyperamy_hits = sum(1 for r in results if r.get('hyperamy', {}).get('hit', False))
    
    print(f'总查询数: {total}')
    print(f'HippoRAG命中: {hipporag_hits}/{total} ({100*hipporag_hits/total:.1f}%)')
    print(f'Fusion命中: {fusion_hits}/{total} ({100*fusion_hits/total:.1f}%)')
    print(f'HyperAmy命中: {hyperamy_hits}/{total} ({100*hyperamy_hits/total:.1f}%)')
PYEOF" 2>/dev/null
        else
            echo "⏳ 实验结果文件尚未生成"
        fi
    else
        echo "🟢 实验正在运行中"
        echo "进程: $(echo "$PROCESS" | awk '{print $2, $11, $12, $13}')"
        echo ""
        
        # 获取最新日志
        echo "【最新日志】"
        LOG_LINES=$(ssh ${SERVER} "cd /public/jiangh/HyperAmy && tail -20 test_three_methods_comparison_monte_cristo.log 2>/dev/null | tail -10" 2>/dev/null)
        
        if [ -n "$LOG_LINES" ]; then
            echo "$LOG_LINES" | while IFS= read -r line; do
                echo "  $line"
            done
        else
            echo "  日志文件为空或不存在"
        fi
        
        echo ""
        echo "【进度提示】"
        echo "  - 如果看到'索引文档'，说明正在索引阶段"
        echo "  - 如果看到'检索'，说明正在检索阶段"
        echo "  - 如果看到进度条，可以查看具体进度"
    fi
    
    echo ""
    echo "=================================================================================="
    echo "下次更新: ${INTERVAL}秒后（按 Ctrl+C 退出）"
    
    sleep ${INTERVAL}
done

