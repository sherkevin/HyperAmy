#!/bin/bash
# 统一监控本地和云端实验

echo "======================================================================"
echo "📊 实验统一监控面板"
echo "======================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 本地实验监控
echo -e "${BLUE}【本地实验】${NC}"
if [ -f test_monte_cristo_comparison.pid ]; then
    PID=$(cat test_monte_cristo_comparison.pid)
    if ps -p $PID > /dev/null 2>&1; then
        ETIME=$(ps -p $PID -o etime= | tr -d ' ')
        PCPU=$(ps -p $PID -o pcpu= | tr -d ' ')
        PMEM=$(ps -p $PID -o pmem= | tr -d ' ')
        echo -e "  ${GREEN}✅ 运行中${NC} (PID: $PID, 已运行: $ETIME, CPU: ${PCPU}%, 内存: ${PMEM}%)"
        
        # 检查进度
        if [ -f test_monte_cristo_comparison.log ]; then
            PROGRESS=$(tail -100 test_monte_cristo_comparison.log | grep -oE "Extracting emotion vectors:.*[0-9]+/[0-9]+" | tail -1)
            if [ -n "$PROGRESS" ]; then
                echo -e "  📈 进度: $PROGRESS"
            fi
        fi
        
        # 检查结果文件
        if [ -f results/monte_cristo_comparison_full.json ]; then
            COUNT=$(python3 -c "import json; print(len(json.load(open('results/monte_cristo_comparison_full.json'))))" 2>/dev/null || echo "0")
            echo -e "  📊 已处理QA对: $COUNT/50"
        fi
    else
        echo -e "  ${RED}❌ 已停止${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠️  未运行${NC}"
fi

echo ""

# 云端实验监控
echo -e "${BLUE}【云端实验】${NC}"
if ssh -o ConnectTimeout=5 jiangh@10.103.16.22 "test -f /media/data4/jiangh/Amygdala/hyperamy_source/test_monte_cristo_comparison_remote.pid" 2>/dev/null; then
    REMOTE_PID=$(ssh -o ConnectTimeout=5 jiangh@10.103.16.22 "cat /media/data4/jiangh/Amygdala/hyperamy_source/test_monte_cristo_comparison_remote.pid" 2>/dev/null)
    if ssh -o ConnectTimeout=5 jiangh@10.103.16.22 "ps -p $REMOTE_PID > /dev/null 2>&1" 2>/dev/null; then
        REMOTE_ETIME=$(ssh -o ConnectTimeout=5 jiangh@10.103.16.22 "ps -p $REMOTE_PID -o etime=" 2>/dev/null | tr -d ' ')
        REMOTE_PCPU=$(ssh -o ConnectTimeout=5 jiangh@10.103.16.22 "ps -p $REMOTE_PID -o pcpu=" 2>/dev/null | tr -d ' ')
        echo -e "  ${GREEN}✅ 运行中${NC} (PID: $REMOTE_PID, 已运行: $REMOTE_ETIME, CPU: ${REMOTE_PCPU}%)"
        
        # 检查日志
        LOG_LINES=$(ssh -o ConnectTimeout=5 jiangh@10.103.16.22 "wc -l < /media/data4/jiangh/Amygdala/hyperamy_source/test_monte_cristo_comparison_remote.log" 2>/dev/null || echo "0")
        if [ "$LOG_LINES" -gt 0 ]; then
            echo -e "  📝 日志: $LOG_LINES 行"
            echo -e "  ${BLUE}最新日志:${NC}"
            ssh -o ConnectTimeout=5 jiangh@10.103.16.22 "tail -3 /media/data4/jiangh/Amygdala/hyperamy_source/test_monte_cristo_comparison_remote.log" 2>/dev/null | sed 's/^/    /'
        else
            echo -e "  ${YELLOW}⚠️  日志为空${NC}"
        fi
    else
        echo -e "  ${RED}❌ 已停止${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠️  未运行或无法连接${NC}"
fi

echo ""
echo "======================================================================"
echo "💡 提示: 运行此脚本可随时查看实验状态"
echo "======================================================================"

