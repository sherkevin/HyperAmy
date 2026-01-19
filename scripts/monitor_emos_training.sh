#!/bin/bash
# emos训练实时监控脚本
# 实时显示训练进度、loss、GPU使用情况等

SERVER_HOST="jiangh@10.103.92.120"
SERVER_PORT="1066"
REMOTE_EMOS_DIR="/public/jiangh/emos"
SCREEN_SESSION="emos-full-training"
UPDATE_INTERVAL=3  # 更新间隔（秒）

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear
echo "=================================================================================="
echo "📊 emos情绪嵌入模型训练 - 实时监控"
echo "=================================================================================="
echo "服务器: $SERVER_HOST:$SERVER_PORT"
echo "项目目录: $REMOTE_EMOS_DIR"
echo "Screen会话: $SCREEN_SESSION"
echo "更新间隔: ${UPDATE_INTERVAL}秒"
echo "💡 提示: 按 Ctrl+C 退出监控"
echo "=================================================================================="
echo ""

start_time=$(date +%s)

while true; do
    clear
    current_time=$(date '+%Y-%m-%d %H:%M:%S')
    elapsed=$(($(date +%s) - start_time))
    hours=$((elapsed / 3600))
    minutes=$(((elapsed % 3600) / 60))
    seconds=$((elapsed % 60))
    
    echo "=================================================================================="
    echo "📊 emos情绪嵌入模型训练 - 实时监控"
    echo "=================================================================================="
    echo "当前时间: $current_time"
    echo "运行时长: ${hours}小时 ${minutes}分钟 ${seconds}秒"
    echo "=================================================================================="
    echo ""
    
    # 检查Screen会话是否存在
    screen_status=$(ssh -p $SERVER_PORT $SERVER_HOST "screen -list 2>/dev/null | grep $SCREEN_SESSION" 2>/dev/null)
    
    if [ -z "$screen_status" ]; then
        echo -e "${RED}❌ Screen会话 '$SCREEN_SESSION' 未运行${NC}"
        echo ""
        echo "💡 提示: 使用以下命令启动训练:"
        echo "   bash scripts/start_full_training.sh"
        echo ""
    else
        echo -e "${GREEN}✅ Screen会话 '$SCREEN_SESSION' 正在运行${NC}"
        echo ""
        
        # 获取最新的训练日志
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📝 最新训练日志（最后15行）:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        latest_log=$(ssh -p $SERVER_PORT $SERVER_HOST "cd $REMOTE_EMOS_DIR && ls -t logs/train_*.log 2>/dev/null | head -1" 2>/dev/null)
        
        if [ -n "$latest_log" ]; then
            ssh -p $SERVER_PORT $SERVER_HOST "cd $REMOTE_EMOS_DIR && tail -15 '$latest_log' 2>/dev/null | tail -15" 2>/dev/null
        else
            echo -e "${YELLOW}⚠️  日志文件尚未生成${NC}"
        fi
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 训练指标摘要:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 提取关键指标
        if [ -n "$latest_log" ]; then
            metrics=$(ssh -p $SERVER_PORT $SERVER_HOST "cd $REMOTE_EMOS_DIR && tail -100 '$latest_log' 2>/dev/null | grep -E '(Epoch|Step|Loss|LR:|Validation)' | tail -5" 2>/dev/null)
            if [ -n "$metrics" ]; then
                echo "$metrics"
            else
                echo -e "${YELLOW}⚠️  等待训练指标输出...${NC}"
            fi
        fi
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "💾 Checkpoint状态:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        checkpoints=$(ssh -p $SERVER_PORT $SERVER_HOST "cd $REMOTE_EMOS_DIR && ls -lht checkpoints/*.pt 2>/dev/null | head -5" 2>/dev/null)
        if [ -n "$checkpoints" ]; then
            echo "$checkpoints"
        else
            echo -e "${YELLOW}⚠️  尚未生成checkpoint文件${NC}"
        fi
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🎮 GPU使用情况:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        gpu_info=$(ssh -p $SERVER_PORT $SERVER_HOST "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1" 2>/dev/null)
        if [ -n "$gpu_info" ]; then
            echo "$gpu_info"
        else
            echo -e "${YELLOW}⚠️  无法获取GPU信息${NC}"
        fi
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💡 常用命令:"
    echo "   • 连接Screen会话: ssh -p $SERVER_PORT $SERVER_HOST -t 'screen -r $SCREEN_SESSION'"
    echo "   • 查看完整日志: ssh -p $SERVER_PORT $SERVER_HOST 'tail -f $REMOTE_EMOS_DIR/logs/train_*.log | tail -1'"
    echo "   • 退出监控: Ctrl+C"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "下次更新: ${UPDATE_INTERVAL}秒后..."
    
    sleep $UPDATE_INTERVAL
done
