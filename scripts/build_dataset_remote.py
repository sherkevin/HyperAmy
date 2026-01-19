#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在远程服务器上构建完整数据集的Python脚本
"""
import sys
import os
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print('=' * 70)
print('构建完整数据集')
print('=' * 70)
print(f'项目路径: {project_root}')
print(f'Python: {sys.executable}')
print(f'开始时间: {os.popen("date").read().strip()}')
print('=' * 70)

# Phase 2: 构建训练集
print('')
print('=' * 70)
print('Phase 2: 构建完整训练集（10000条）')
print('=' * 70)
result = subprocess.run([
    sys.executable, 
    'src/build_training_set.py',
    '--input', 'data/books/monte_cristo_clean.txt',
    '--output', 'data/training/monte_cristo_train_full.jsonl',
    '--max-sentences', '10000',
    '--batch-size', '100'
], cwd=str(project_root), timeout=7200, env=dict(os.environ, PYTHONUNBUFFERED='1'))  # 2小时超时，无缓冲输出

if result.returncode != 0:
    print(f'❌ Phase 2失败，退出码: {result.returncode}')
    sys.exit(1)

# Phase 3: 生成QA
print('')
print('=' * 70)
print('Phase 3: 生成完整QA基准测试（50条）')
print('=' * 70)
result = subprocess.run([
    sys.executable,
    'src/gen_public_qa.py',
    '--chunks', 'data/training/monte_cristo_train_full.jsonl',
    '--output', 'data/public_benchmark/monte_cristo_qa_full.json',
    '--num', '50',
    '--top-k', '100'
], cwd=str(project_root), env=dict(os.environ, PYTHONUNBUFFERED='1'))

if result.returncode != 0:
    print(f'❌ Phase 3失败，退出码: {result.returncode}')
    sys.exit(1)

print('')
print('=' * 70)
print('✅ 数据集构建完成！')
print(f'结束时间: {os.popen("date").read().strip()}')
print('=' * 70)

