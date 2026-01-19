#!/bin/bash
# SSH连接监控脚本
# 定期检查SSH连接是否恢复

SERVER="10.103.92.120"
PORT="1066"
USER="jiangh"
CHECK_INTERVAL=300  # 每5分钟检查一次（秒）

echo "开始监控SSH连接：$USER@$SERVER:$PORT"
echo "检查间隔：${CHECK_INTERVAL}秒（5分钟）"
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 测试ping
    if ping -c 1 -W 1 $SERVER >/dev/null 2>&1; then
        PING_STATUS="✅"
    else
        PING_STATUS="❌"
    fi
    
    # 测试端口
    if nc -zv -G 2 $SERVER $PORT >/dev/null 2>&1; then
        PORT_STATUS="✅"
    else
        PORT_STATUS="❌"
    fi
    
    # 测试SSH连接
    SSH_OUTPUT=$(ssh -F /dev/null -o ConnectTimeout=5 -o BatchMode=yes -p $PORT $USER@$SERVER "echo OK" 2>&1)
    if [ $? -eq 0 ]; then
        SSH_STATUS="✅ 连接成功！"
        echo ""
        echo "============================================================"
        echo "🎉 SSH连接已恢复！"
        echo "时间：$TIMESTAMP"
        echo "============================================================"
        echo ""
        echo "可以尝试连接："
        echo "ssh -p $PORT $USER@$SERVER"
        echo ""
        break
    else
        SSH_STATUS="❌ 连接失败"
    fi
    
    echo "[$TIMESTAMP] Ping: $PING_STATUS | 端口: $PORT_STATUS | SSH: $SSH_STATUS"
    
    sleep $CHECK_INTERVAL
done
