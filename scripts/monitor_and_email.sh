#!/bin/bash
# 监控实验并在完成后发送邮件

REMOTE_HOST="jiangh@10.103.16.22"
REMOTE_PATH="/media/data4/jiangh/Amygdala/hyperamy_source"
RESULTS_DIR="$REMOTE_PATH/outputs/rigorous_experiment"
CHECK_INTERVAL=300  # 每5分钟检查一次（秒）
MAX_WAIT_HOURS=6    # 最多等待6小时

echo "=========================================="
echo "监控实验并在完成后发送邮件"
echo "=========================================="
echo "检查间隔: $CHECK_INTERVAL 秒 (5分钟)"
echo "最长等待: $MAX_WAIT_HOURS 小时"
echo ""

# 检查实验是否完成
check_experiment_complete() {
    ssh "$REMOTE_HOST" "test -f $RESULTS_DIR/comparison_results.json" 2>/dev/null
    return $?
}

# 检查实验是否在运行
check_experiment_running() {
    ssh "$REMOTE_HOST" "ps aux | grep test_rigorous_comparison_save | grep -v grep" > /dev/null 2>&1
    return $?
}

# 等待实验完成
wait_count=0
max_waits=$((MAX_WAIT_HOURS * 3600 / CHECK_INTERVAL))

echo "开始监控实验..."
while [ $wait_count -lt $max_waits ]; do
    # 检查是否完成
    if check_experiment_complete; then
        echo ""
        echo "✅ 实验已完成！"
        break
    fi
    
    # 检查是否还在运行
    if ! check_experiment_running; then
        echo ""
        echo "⚠️  实验进程不存在，但结果文件未生成"
        echo "检查日志..."
        ssh "$REMOTE_HOST" "cd $REMOTE_PATH && tail -20 outputs/rigorous_experiment/logs/experiment_*.log 2>/dev/null | tail -10"
        break
    fi
    
    # 显示进度
    elapsed=$((wait_count * CHECK_INTERVAL / 60))
    echo -ne "\r等待中... (已等待 ${elapsed} 分钟)"
    
    sleep $CHECK_INTERVAL
    wait_count=$((wait_count + 1))
done

echo ""
echo ""

# 下载结果文件到本地
echo "【1】下载实验结果..."
LOCAL_RESULTS_DIR="./outputs/rigorous_experiment_remote"
mkdir -p "$LOCAL_RESULTS_DIR"

rsync -avz "$REMOTE_HOST:$RESULTS_DIR/" "$LOCAL_RESULTS_DIR/" || {
    echo "⚠️  下载结果失败，尝试直接发送"
}

# 发送邮件
echo ""
echo "【2】发送邮件通知..."
python3 "$(dirname "$0")/send_experiment_email.py" "$LOCAL_RESULTS_DIR" || {
    echo "⚠️  邮件发送失败，但结果已下载到: $LOCAL_RESULTS_DIR"
}

echo ""
echo "=========================================="
echo "✅ 监控完成"
echo "=========================================="
echo "结果文件位置: $LOCAL_RESULTS_DIR"

