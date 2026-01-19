#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查服务器硬件资源并评估可部署的模型

分析服务器资源，推荐适合部署的开源大模型（推理和训练）
"""

import subprocess
import sys
from typing import Dict, List, Tuple
import json


def run_ssh_command(host: str, port: int, user: str, command: str) -> Tuple[str, int]:
    """运行SSH命令"""
    cmd = f"ssh -p {port} {user}@{host} '{command}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode


def get_gpu_info(host: str, port: int, user: str) -> List[Dict]:
    """获取GPU信息"""
    command = "nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader"
    output, code = run_ssh_command(host, port, user, command)
    
    gpus = []
    if code == 0 and output:
        for line in output.split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                try:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_total_gb': int(parts[2].replace(' MiB', '')) / 1024,
                        'memory_free_gb': int(parts[3].replace(' MiB', '')) / 1024,
                        'compute_capability': parts[4]
                    })
                except (ValueError, IndexError):
                    continue
    return gpus


def get_system_info(host: str, port: int, user: str) -> Dict:
    """获取系统信息"""
    info = {}
    
    # CPU信息
    cmd = "lscpu | grep -E '^CPU\\(|^Thread|^Core|^Socket|^Model name' | head -5"
    output, _ = run_ssh_command(host, port, user, cmd)
    info['cpu_info'] = output
    
    # 内存信息
    cmd = "free -h | grep '^Mem:'"
    output, _ = run_ssh_command(host, port, user, cmd)
    info['memory_info'] = output
    
    # 磁盘信息
    cmd = "df -h /public 2>/dev/null | tail -1 || df -h / | tail -1"
    output, _ = run_ssh_command(host, port, user, cmd)
    info['disk_info'] = output
    
    return info


def estimate_model_size(model_name: str, params_b: float) -> Dict:
    """估算模型大小"""
    # FP32: 4 bytes per parameter
    # FP16/BF16: 2 bytes per parameter
    # INT8: 1 byte per parameter
    
    return {
        'fp32_gb': params_b * 4 / 1024**3,
        'fp16_gb': params_b * 2 / 1024**3,
        'int8_gb': params_b * 1 / 1024**3,
        'params_billions': params_b
    }


def recommend_models_for_inference(gpu_memory_gb: float, num_gpus: int) -> List[Dict]:
    """推荐适合推理的模型"""
    recommendations = []
    
    # 常用开源模型
    models = [
        {'name': 'Llama-2-7B', 'params': 7, 'min_memory_fp16': 14, 'min_memory_int8': 7},
        {'name': 'Llama-2-13B', 'params': 13, 'min_memory_fp16': 26, 'min_memory_int8': 13},
        {'name': 'Llama-2-70B', 'params': 70, 'min_memory_fp16': 140, 'min_memory_int8': 70},
        {'name': 'Qwen-7B', 'params': 7, 'min_memory_fp16': 14, 'min_memory_int8': 7},
        {'name': 'Qwen-14B', 'params': 14, 'min_memory_fp16': 28, 'min_memory_int8': 14},
        {'name': 'ChatGLM-6B', 'params': 6, 'min_memory_fp16': 12, 'min_memory_int8': 6},
        {'name': 'Baichuan-7B', 'params': 7, 'min_memory_fp16': 14, 'min_memory_int8': 7},
        {'name': 'InternLM-7B', 'params': 7, 'min_memory_fp16': 14, 'min_memory_int8': 7},
        {'name': 'Mistral-7B', 'params': 7, 'min_memory_fp16': 14, 'min_memory_int8': 7},
        {'name': 'Yi-6B', 'params': 6, 'min_memory_fp16': 12, 'min_memory_int8': 6},
    ]
    
    total_memory = gpu_memory_gb * num_gpus
    
    for model in models:
        can_run_fp16 = gpu_memory_gb >= model['min_memory_fp16'] * 0.8  # 80%阈值
        can_run_int8 = gpu_memory_gb >= model['min_memory_int8'] * 0.8
        can_run_multi_gpu = total_memory >= model['min_memory_fp16'] * 0.8
        
        if can_run_fp16 or can_run_int8 or (num_gpus > 1 and can_run_multi_gpu):
            rec = {
                'model': model['name'],
                'params': f"{model['params']}B",
                'single_gpu_fp16': '✅' if can_run_fp16 else '❌',
                'single_gpu_int8': '✅' if can_run_int8 else '❌',
                'multi_gpu_fp16': '✅' if (num_gpus > 1 and can_run_multi_gpu) else '❌',
            }
            recommendations.append(rec)
    
    return recommendations


def recommend_models_for_training(gpu_memory_gb: float, num_gpus: int) -> List[Dict]:
    """推荐适合训练的模型"""
    recommendations = []
    
    # 训练需要更多内存（需要存储梯度、优化器状态等）
    # 通常需要3-4倍推理内存
    models = [
        {'name': 'Llama-2-7B', 'params': 7, 'min_memory_per_gpu': 40},
        {'name': 'Qwen-7B', 'params': 7, 'min_memory_per_gpu': 40},
        {'name': 'ChatGLM-6B', 'params': 6, 'min_memory_per_gpu': 35},
        {'name': 'Baichuan-7B', 'params': 7, 'min_memory_per_gpu': 40},
        {'name': 'InternLM-7B', 'params': 7, 'min_memory_per_gpu': 40},
    ]
    
    for model in models:
        can_train_single = gpu_memory_gb >= model['min_memory_per_gpu']
        can_train_multi = (gpu_memory_gb * num_gpus) >= model['min_memory_per_gpu'] * 2
        
        if can_train_single or (num_gpus >= 2 and can_train_multi):
            rec = {
                'model': model['name'],
                'params': f"{model['params']}B",
                'single_gpu': '✅' if can_train_single else '❌',
                'multi_gpu': '✅' if (num_gpus >= 2 and can_train_multi) else '❌',
                'recommended_gpus': min(4, num_gpus) if can_train_multi else 1,
            }
            recommendations.append(rec)
    
    return recommendations


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="检查服务器硬件资源并评估可部署的模型")
    parser.add_argument('--host', type=str, default='10.103.92.120', help='服务器地址')
    parser.add_argument('--port', type=int, default=1066, help='SSH端口')
    parser.add_argument('--user', type=str, default='jiangh', help='用户名')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"检查服务器硬件资源: {args.user}@{args.host}:{args.port}")
    print("=" * 80)
    
    # 获取GPU信息
    print("\n【GPU信息】")
    gpus = get_gpu_info(args.host, args.port, args.user)
    if gpus:
        total_memory = 0
        for gpu in gpus:
            print(f"GPU {gpu['index']}: {gpu['name']}")
            print(f"  显存: {gpu['memory_total_gb']:.1f}GB (空闲: {gpu['memory_free_gb']:.1f}GB)")
            print(f"  Compute Capability: {gpu['compute_capability']}")
            total_memory += gpu['memory_total_gb']
        
        avg_memory = total_memory / len(gpus)
        print(f"\n总计: {len(gpus)}个GPU, 平均显存: {avg_memory:.1f}GB/GPU, 总显存: {total_memory:.1f}GB")
    else:
        print("❌ 无法获取GPU信息")
        return
    
    # 获取系统信息
    print("\n【系统信息】")
    sys_info = get_system_info(args.host, args.port, args.user)
    if sys_info.get('cpu_info'):
        print("CPU:")
        print(sys_info['cpu_info'])
    if sys_info.get('memory_info'):
        print(f"\n内存: {sys_info['memory_info']}")
    if sys_info.get('disk_info'):
        print(f"\n磁盘: {sys_info['disk_info']}")
    
    # 推理推荐
    print("\n" + "=" * 80)
    print("【推理模型推荐】")
    print("=" * 80)
    inference_recs = recommend_models_for_inference(avg_memory, len(gpus))
    print(f"{'模型':<20} {'参数':<10} {'单GPU FP16':<15} {'单GPU INT8':<15} {'多GPU FP16':<15}")
    print("-" * 80)
    for rec in inference_recs:
        print(f"{rec['model']:<20} {rec['params']:<10} {rec['single_gpu_fp16']:<15} {rec['single_gpu_int8']:<15} {rec['multi_gpu_fp16']:<15}")
    
    # 训练推荐
    print("\n" + "=" * 80)
    print("【训练模型推荐】")
    print("=" * 80)
    training_recs = recommend_models_for_training(avg_memory, len(gpus))
    print(f"{'模型':<20} {'参数':<10} {'单GPU训练':<15} {'多GPU训练':<15} {'推荐GPU数':<15}")
    print("-" * 80)
    for rec in training_recs:
        print(f"{rec['model']:<20} {rec['params']:<10} {rec['single_gpu']:<15} {rec['multi_gpu']:<15} {rec['recommended_gpus']:<15}")
    
    # 总结
    print("\n" + "=" * 80)
    print("【总结】")
    print("=" * 80)
    print(f"✅ 单GPU推理: 可运行 {len([r for r in inference_recs if '✅' in r['single_gpu_fp16'] or '✅' in r['single_gpu_int8']])} 个模型（FP16或INT8）")
    print(f"✅ 多GPU推理: 可运行更大模型（使用 {len(gpus)} 个GPU）")
    print(f"✅ 单GPU训练: 可训练 {len([r for r in training_recs if '✅' in r['single_gpu']])} 个7B级模型")
    print(f"✅ 多GPU训练: 可训练 {len([r for r in training_recs if '✅' in r['multi_gpu']])} 个模型（使用多GPU）")
    
    print("\n💡 建议:")
    print("  1. 推理优先使用FP16或INT8量化，节省显存")
    print("  2. 训练建议使用多GPU（DeepSpeed/FSDP）")
    print("  3. 使用vLLM/Text Generation Inference加速推理")
    print("  4. 使用LoRA/PEFT进行高效微调（节省显存）")


if __name__ == "__main__":
    main()
