#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Qwen3-Embedding-8B模型下载速度
"""
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor

def test_endpoint_speed(endpoint_name, base_url, test_file_url):
    """测试下载速度"""
    try:
        start_time = time.time()
        response = requests.get(test_file_url, timeout=10, stream=True)
        if response.status_code == 200:
            # 读取前1MB数据
            data = response.raw.read(1024 * 1024)
            elapsed = time.time() - start_time
            speed = len(data) / elapsed / (1024 * 1024)  # MB/s
            return {
                'endpoint': endpoint_name,
                'url': base_url,
                'speed_mbps': speed,
                'latency': elapsed,
                'status': 'success'
            }
        else:
            return {
                'endpoint': endpoint_name,
                'url': base_url,
                'speed_mbps': 0,
                'latency': 0,
                'status': f'failed: {response.status_code}'
            }
    except Exception as e:
        return {
            'endpoint': endpoint_name,
            'url': base_url,
            'speed_mbps': 0,
            'latency': 0,
            'status': f'error: {str(e)[:50]}'
        }

def main():
    print("="*70)
    print("测试Qwen3-Embedding-8B模型下载速度")
    print("="*70)
    
    # 测试文件（使用模型的一个小文件）
    test_file = "config.json"
    
    # 不同的下载源
    endpoints = [
        {
            'name': 'HF官方',
            'base_url': 'https://huggingface.co',
            'test_url': f'https://huggingface.co/Qwen/Qwen3-Embedding-8B/resolve/main/{test_file}'
        },
        {
            'name': 'HF镜像（hf-mirror）',
            'base_url': 'https://hf-mirror.com',
            'test_url': f'https://hf-mirror.com/Qwen/Qwen3-Embedding-8B/resolve/main/{test_file}'
        },
    ]
    
    print("\n测试各下载源速度...")
    results = []
    
    for endpoint in endpoints:
        print(f"\n测试: {endpoint['name']}...")
        result = test_endpoint_speed(
            endpoint['name'],
            endpoint['base_url'],
            endpoint['test_url']
        )
        results.append(result)
        if result['status'] == 'success':
            print(f"  ✅ 速度: {result['speed_mbps']:.2f} MB/s, 延迟: {result['latency']:.2f}s")
        else:
            print(f"  ❌ {result['status']}")
    
    # 选择最快的
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        best = max(successful, key=lambda x: x['speed_mbps'])
        print(f"\n✅ 最快下载源: {best['endpoint']} ({best['speed_mbps']:.2f} MB/s)")
        print(f"   推荐使用: {best['url']}")
        return best['url']
    else:
        print("\n❌ 所有下载源测试失败")
        return None

if __name__ == "__main__":
    main()
