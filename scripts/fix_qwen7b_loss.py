"""
修复Qwen-7B的Cal Loss过大问题
方案：降低lambda_cal权重，调整alpha_scale
"""
import os
import sys

EMOS_PATH = os.environ.get('EMOS_PATH', 'emos-master')
CONFIG_FILE = os.path.join(EMOS_PATH, 'src', 'config.py')

print("="*70)
print("修复Qwen-7B Cal Loss过大问题")
print("="*70)

# 读取配置文件
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

print("\n当前配置:")
if 'lambda_cal: float = ' in content:
    import re
    match = re.search(r'lambda_cal: float = ([0-9.]+)', content)
    if match:
        print(f"  lambda_cal = {match.group(1)}")
if 'alpha_scale: float = ' in content:
    match = re.search(r'alpha_scale: float = ([0-9.]+)', content)
    if match:
        print(f"  alpha_scale = {match.group(1)}")

print("\n建议调整:")
print("  1. lambda_cal: 0.1 -> 0.01 (降低10倍)")
print("  2. alpha_scale: 50.0 -> 30.0 (降低40%)")
print("\n原因:")
print("  - Cal Loss = MSE(kappa_pred, kappa_target)")
print("  - kappa_target = 1.0 + alpha_scale * max(soft_label)")
print("  - 降低lambda_cal可以减少cal loss对总loss的影响")
print("  - 降低alpha_scale可以减小target_kappa的值，使预测更容易对齐")

print("\n" + "="*70)
print("需要修改 config.py 吗？(y/n)")

