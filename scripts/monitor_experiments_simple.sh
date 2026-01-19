#!/bin/bash
# 简单的监控脚本 - 通过SSH查看实时进度

SERVER="hyperamy-server"
INTERVAL=3  # 更新间隔（秒）

echo "=================================================================================="
echo "🔍 实验实时监控（按 Ctrl+C 退出）"
echo "=================================================================================="
echo "更新间隔: ${INTERVAL}秒"
echo "服务器: ${SERVER}"
echo "=================================================================================="
echo ""

while true; do
    clear
    echo "=================================================================================="
    echo "🔍 实验实时监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================================="
    echo ""
    
    # 第一批实验状态
    echo "📊 第一批实验 (V1 - 原始版本)"
    echo "--------------------------------------------------------------------------------"
    
    # 获取进度条
    PROGRESS=$(ssh ${SERVER} "cd /public/jiangh/HyperAmy && tail -50 test_two_methods_comparison.log 2>/dev/null | grep 'Extracting emotion vectors' | tail -1" 2>/dev/null)
    
    if [ -n "$PROGRESS" ]; then
        echo "状态: 🟢 正在运行"
        echo "进度: $PROGRESS"
        
        # 提取百分比
        PERCENT=$(echo "$PROGRESS" | grep -oP '\d+(?=%)' | tail -1)
        if [ -n "$PERCENT" ]; then
            CURRENT=$(echo "$PERCENT * 468 / 100" | bc | cut -d. -f1)
            REMAINING=$((468 - CURRENT))
            echo "完成度: ${CURRENT}/468 (${PERCENT}%)"
        fi
    else
        # 检查是否完成
        LAST_LINES=$(ssh ${SERVER} "cd /public/jiangh/HyperAmy && tail -5 test_two_methods_comparison.log 2>/dev/null | tail -3" 2>/dev/null)
        if echo "$LAST_LINES" | grep -q "实验完成\|已完成\|完成"; then
            echo "状态: ✅ 已完成"
        elif [ -n "$LAST_LINES" ]; then
            echo "状态: 🟢 运行中"
            echo "最新日志: $(echo "$LAST_LINES" | tail -1 | cut -c1-80)"
        else
            echo "状态: ⏳ 等待中..."
        fi
    fi
    
    # 检查进程
    PROCESS=$(ssh ${SERVER} "ps aux | grep '[t]est_two_methods_comparison.py' | grep -v v2 | head -1" 2>/dev/null)
    if [ -z "$PROCESS" ]; then
        echo "⚠️  进程未运行"
    fi
    
    echo ""
    
    # 第二批实验状态
    echo "📊 第二批实验 (V2 - 优化版本)"
    echo "--------------------------------------------------------------------------------"
    
    # 检查结果文件
    RESULT_EXISTS=$(ssh ${SERVER} "test -f /public/jiangh/HyperAmy/outputs/two_methods_comparison_v2/comparison_results.json && echo 'yes' || echo 'no'" 2>/dev/null)
    
    if [ "$RESULT_EXISTS" = "yes" ]; then
        echo "状态: ✅ 已完成"
        
        # 获取结果统计
        STATS=$(ssh ${SERVER} "cd /public/jiangh/HyperAmy && python3 << 'PYEOF'
import json
try:
    with open('outputs/two_methods_comparison_v2/comparison_results.json') as f:
        results = json.load(f)
    print(f'结果数: {len(results)}')
    hipporag_ok = sum(1 for r in results if r.get('hipporag', {}).get('available', False))
    fusion_ok = sum(1 for r in results if r.get('fusion', {}).get('available', False))
    print(f'HippoRAG成功: {hipporag_ok}/{len(results)}')
    print(f'Fusion成功: {fusion_ok}/{len(results)}')
except Exception as e:
    print(f'读取结果失败: {e}')
PYEOF" 2>/dev/null)
        
        if [ -n "$STATS" ]; then
            echo "$STATS" | sed 's/^/  /'
        fi
    else
        # 检查日志
        V2_LOG=$(ssh ${SERVER} "cd /public/jiangh/HyperAmy && tail -5 test_two_methods_comparison_v2.log 2>/dev/null | tail -3" 2>/dev/null)
        if [ -n "$V2_LOG" ]; then
            echo "状态: ⏳ 运行中"
            echo "最新日志: $(echo "$V2_LOG" | tail -1 | cut -c1-80)"
        else
            echo "状态: ⏳ 未开始或状态未知"
        fi
    fi
    
    echo ""
    echo "=================================================================================="
    echo "下次更新: ${INTERVAL}秒后（按 Ctrl+C 退出）"
    
    sleep ${INTERVAL}
done

