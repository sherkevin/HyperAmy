#!/bin/bash
# 监控重试进度

REMOTE_HOST="jiangh@10.103.16.22"
REMOTE_PATH="/media/data4/jiangh/Amygdala/hyperamy_source"

echo "======================================================================"
echo "监控重试进度"
echo "======================================================================"

ssh "$REMOTE_HOST" "cd $REMOTE_PATH && \
  echo '【1】进程状态:' && \
  if [ -f results/retry.pid ]; then
    PID=\$(cat results/retry.pid)
    if ps -p \$PID > /dev/null 2>&1; then
      echo '  ✅ 进程运行中 (PID: '\$PID')'
      ps -p \$PID -o pid,etime,command
    else
      echo '  ❌ 进程已结束'
    fi
  else
    echo '  ⚠️  PID文件不存在'
  fi && \
  echo '' && \
  echo '【2】最新日志（最后20行）:' && \
  if [ -f results/retry.log ]; then
    tail -20 results/retry.log
  else
    echo '  日志文件不存在'
  fi && \
  echo '' && \
  echo '【3】结果文件状态:' && \
  if [ -f results/experiment_full_retried.json ]; then
    echo '  ✅ 结果文件存在'
    python3 << 'PYEOF'
import json
import os
try:
    with open('results/experiment_full_retried.json', 'r') as f:
        results = json.load(f)
    total = len(results)
    valid = sum(1 for r in results 
               if '出错' not in r.get('oracle', {}).get('answer', '') and
                  '出错' not in r.get('baseline', {}).get('answer', '') and
                  '出错' not in r.get('hyperamy', {}).get('answer', ''))
    print(f'  总问题数: {total}')
    print(f'  有效结果: {valid} ({100*valid/total:.1f}%)')
    print(f'  文件大小: {os.path.getsize(\"results/experiment_full_retried.json\")/1024:.1f} KB')
except Exception as e:
    print(f'  读取错误: {e}')
PYEOF
  else
    echo '  ⏳ 结果文件尚未创建'
  fi"

echo ""
echo "======================================================================"
echo "持续监控命令:"
echo "  watch -n 10 './monitor_retry.sh'"
echo "  或: ssh $REMOTE_HOST 'cd $REMOTE_PATH && tail -f results/retry.log'"
echo "======================================================================"

