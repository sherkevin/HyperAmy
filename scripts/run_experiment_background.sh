#!/bin/bash
# 后台运行严谨对比实验

# 进入项目目录
cd /Users/ginger/Desktop/学术教育/科研计划/nips/HyperAmy

# 创建日志目录
mkdir -p outputs/rigorous_experiment/logs

# 使用 nohup 在后台运行，输出重定向到日志文件
nohup python -m test.test_rigorous_comparison_save > outputs/rigorous_experiment/logs/experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 获取进程ID
PID=$!

# 保存进程ID
echo $PID > outputs/rigorous_experiment/experiment.pid

echo "✅ 实验已在后台启动"
echo "进程ID: $PID"
echo "日志文件: outputs/rigorous_experiment/logs/experiment_*.log"
echo ""
echo "查看进度:"
echo "  tail -f outputs/rigorous_experiment/logs/experiment_*.log"
echo ""
echo "检查状态:"
echo "  ps -p $PID"
echo ""
echo "停止实验:"
echo "  kill $PID"

