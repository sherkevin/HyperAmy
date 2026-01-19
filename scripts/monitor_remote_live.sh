#!/bin/bash
# 实时动态监控云端实验

echo "======================================================================"
echo "📊 云端实验实时监控"
echo "======================================================================"
echo "按 Ctrl+C 退出监控"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 监控循环
while true; do
    clear
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${CYAN}📊 云端实验实时监控 - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    
    # 检查进程状态
    echo -e "${BLUE}【进程状态】${NC}"
    REMOTE_STATUS=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no jiangh@10.103.16.22 << 'EOF' 2>/dev/null
cd /media/data4/jiangh/Amygdala/hyperamy_source
if [ -f test_monte_cristo_comparison_remote.pid ]; then
    PID=$(cat test_monte_cristo_comparison_remote.pid)
    if ps -p $PID > /dev/null 2>&1; then
        ETIME=$(ps -p $PID -o etime= | tr -d ' ')
        PCPU=$(ps -p $PID -o pcpu= | tr -d ' ')
        PMEM=$(ps -p $PID -o pmem= | tr -d ' ')
        STATE=$(ps -p $PID -o state= | tr -d ' ')
        echo "PID=$PID|ETIME=$ETIME|PCPU=$PCPU|PMEM=$PMEM|STATE=$STATE"
    else
        echo "STOPPED"
    fi
else
    echo "NOT_RUNNING"
fi
EOF
)
    
    if [ -z "$REMOTE_STATUS" ]; then
        echo -e "  ${RED}❌ 无法连接到云端服务器${NC}"
    elif [ "$REMOTE_STATUS" = "NOT_RUNNING" ]; then
        echo -e "  ${YELLOW}⚠️  实验未运行${NC}"
    elif [ "$REMOTE_STATUS" = "STOPPED" ]; then
        echo -e "  ${RED}❌ 进程已停止${NC}"
    else
        PID=$(echo $REMOTE_STATUS | sed 's/.*PID=\([^|]*\).*/\1/')
        ETIME=$(echo $REMOTE_STATUS | sed 's/.*ETIME=\([^|]*\).*/\1/')
        PCPU=$(echo $REMOTE_STATUS | sed 's/.*PCPU=\([^|]*\).*/\1/')
        PMEM=$(echo $REMOTE_STATUS | sed 's/.*PMEM=\([^|]*\).*/\1/')
        STATE=$(echo $REMOTE_STATUS | sed 's/.*STATE=\([^|]*\).*/\1/')
        
        echo -e "  ${GREEN}✅ 运行中${NC}"
        echo -e "     PID: $PID"
        echo -e "     已运行: $ETIME"
        echo -e "     CPU: ${PCPU}%"
        echo -e "     内存: ${PMEM}%"
        echo -e "     状态: $STATE"
    fi
    
    echo ""
    
    # 检查日志
    echo -e "${BLUE}【日志状态】${NC}"
    LOG_INFO=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no jiangh@10.103.16.22 << 'EOF' 2>/dev/null
cd /media/data4/jiangh/Amygdala/hyperamy_source
if [ -f test_monte_cristo_comparison_remote.log ]; then
    LINES=$(wc -l < test_monte_cristo_comparison_remote.log)
    SIZE=$(wc -c < test_monte_cristo_comparison_remote.log)
    echo "LINES=$LINES|SIZE=$SIZE"
else
    echo "NO_FILE"
fi
EOF
)
    
    if [ -z "$LOG_INFO" ]; then
        echo -e "  ${RED}❌ 无法获取日志信息${NC}"
    elif [ "$LOG_INFO" = "NO_FILE" ]; then
        echo -e "  ${YELLOW}⚠️  日志文件不存在${NC}"
    else
        LINES=$(echo $LOG_INFO | sed 's/.*LINES=\([^|]*\).*/\1/')
        SIZE=$(echo $LOG_INFO | sed 's/.*SIZE=\([^|]*\).*/\1/')
        SIZE_KB=$((SIZE / 1024))
        echo -e "  ${GREEN}✅ 日志文件存在${NC}"
        echo -e "     行数: $LINES"
        echo -e "     大小: ${SIZE_KB} KB"
    fi
    
    echo ""
    
    # 显示最新日志
    echo -e "${BLUE}【最新日志（最后20行）】${NC}"
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no jiangh@10.103.16.22 << 'EOF' 2>/dev/null | grep -v "^Welcome\|^ \*\|^   -\|^406\|^333\|^New\|^Your" | sed 's/^/  /'
cd /media/data4/jiangh/Amygdala/hyperamy_source
if [ -f test_monte_cristo_comparison_remote.log ] && [ -s test_monte_cristo_comparison_remote.log ]; then
    tail -20 test_monte_cristo_comparison_remote.log
else
    echo "日志为空或不存在"
fi
EOF
    
    echo ""
    
    # 检查结果文件
    echo -e "${BLUE}【结果文件】${NC}"
    RESULT_INFO=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no jiangh@10.103.16.22 << 'EOF' 2>/dev/null
cd /media/data4/jiangh/Amygdala/hyperamy_source
if [ -f results/monte_cristo_comparison_remote.json ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('results/monte_cristo_comparison_remote.json'))))" 2>/dev/null || echo "0")
    SIZE=$(wc -c < results/monte_cristo_comparison_remote.json)
    echo "COUNT=$COUNT|SIZE=$SIZE"
else
    echo "NO_FILE"
fi
EOF
)
    
    if [ -z "$RESULT_INFO" ]; then
        echo -e "  ${YELLOW}⚠️  无法检查结果文件${NC}"
    elif [ "$RESULT_INFO" = "NO_FILE" ]; then
        echo -e "  ${YELLOW}⚠️  结果文件尚未创建${NC}"
    else
        COUNT=$(echo $RESULT_INFO | sed 's/.*COUNT=\([^|]*\).*/\1/')
        SIZE=$(echo $RESULT_INFO | sed 's/.*SIZE=\([^|]*\).*/\1/')
        SIZE_KB=$((SIZE / 1024))
        echo -e "  ${GREEN}✅ 结果文件存在${NC}"
        echo -e "     已处理QA对: $COUNT/50"
        echo -e "     文件大小: ${SIZE_KB} KB"
    fi
    
    echo ""
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${CYAN}刷新间隔: 5秒 | 按 Ctrl+C 退出${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    
    sleep 5
done

