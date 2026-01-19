#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试不同HF镜像的下载速度，选择最快的源

测试的镜像：
1. 官方源：https://huggingface.co
2. HF镜像（中国）：https://hf-mirror.com
3. HF镜像（备用）：https://hf-mirror.com
"""
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 测试用的模型文件URL（选择一个较小的文件）
TEST_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TEST_FILE = "tokenizer.json"  # 较小文件，用于测试速度

print("=" * 70)
print("测试 HuggingFace 下载速度")
print("=" * 70)
print(f"测试模型: {TEST_MODEL}")
print(f"测试文件: {TEST_FILE}")
print("=" * 70)

# 测试的镜像列表
mirrors = [
    {
        "name": "官方源 (huggingface.co)",
        "base_url": "https://huggingface.co",
        "use_mirror": False
    },
    {
        "name": "HF镜像 (hf-mirror.com)",
        "base_url": "https://hf-mirror.com",
        "use_mirror": True
    },
]


def test_download_speed(mirror_info, timeout=10):
    """测试单个镜像的下载速度"""
    name = mirror_info["name"]
    base_url = mirror_info["base_url"]
    
    # 构造下载URL
    file_url = f"{base_url}/{TEST_MODEL}/resolve/main/{TEST_FILE}"
    
    print(f"\n测试: {name}")
    print(f"  URL: {file_url}")
    
    try:
        start_time = time.time()
        
        # 发送HEAD请求测试连接
        response = requests.head(file_url, timeout=timeout, allow_redirects=True)
        head_time = time.time() - start_time
        
        if response.status_code != 200:
            # 如果HEAD不支持，尝试GET（只下载一小部分）
            response = requests.get(
                file_url, 
                timeout=timeout, 
                stream=True,
                headers={"Range": "bytes=0-1024"}  # 只下载前1KB
            )
        
        # 如果支持Range，下载1KB测试
        if response.status_code in [200, 206]:
            data_size = len(response.content)
            download_time = time.time() - start_time
            
            # 估算速度（MB/s）
            if download_time > 0:
                speed_mbps = (data_size / 1024 / 1024) / download_time
                latency_ms = head_time * 1000
                
                result = {
                    "name": name,
                    "base_url": base_url,
                    "use_mirror": mirror_info["use_mirror"],
                    "success": True,
                    "latency_ms": latency_ms,
                    "speed_mbps": speed_mbps,
                    "download_time": download_time,
                    "status_code": response.status_code
                }
                
                print(f"  ✅ 成功")
                print(f"  延迟: {latency_ms:.1f}ms")
                print(f"  速度: {speed_mbps:.2f} MB/s")
                return result
            else:
                raise ValueError("下载时间为0")
        else:
            raise ValueError(f"HTTP {response.status_code}")
            
    except requests.exceptions.Timeout:
        print(f"  ❌ 超时（>{timeout}秒）")
        return {
            "name": name,
            "base_url": base_url,
            "use_mirror": mirror_info["use_mirror"],
            "success": False,
            "error": "Timeout"
        }
    except requests.exceptions.ConnectionError as e:
        print(f"  ❌ 连接失败: {str(e)[:50]}")
        return {
            "name": name,
            "base_url": base_url,
            "use_mirror": mirror_info["use_mirror"],
            "success": False,
            "error": "ConnectionError"
        }
    except Exception as e:
        print(f"  ❌ 错误: {str(e)[:50]}")
        return {
            "name": name,
            "base_url": base_url,
            "use_mirror": mirror_info["use_mirror"],
            "success": False,
            "error": str(e)[:50]
        }


def test_all_mirrors():
    """测试所有镜像"""
    results = []
    
    for mirror in mirrors:
        result = test_download_speed(mirror)
        results.append(result)
        time.sleep(1)  # 避免请求过快
    
    return results


def find_best_mirror(results):
    """找出最快的镜像"""
    successful_results = [r for r in results if r.get("success", False)]
    
    if not successful_results:
        print("\n❌ 所有镜像测试失败，请检查网络连接")
        return None
    
    # 按速度排序（优先考虑速度，其次考虑延迟）
    best = max(
        successful_results, 
        key=lambda x: (x.get("speed_mbps", 0), -x.get("latency_ms", 9999))
    )
    
    return best


# 运行测试
print("\n开始测试...")
results = test_all_mirrors()

# 显示结果
print("\n" + "=" * 70)
print("测试结果汇总")
print("=" * 70)

for result in results:
    if result.get("success"):
        print(f"\n✅ {result['name']}")
        print(f"   延迟: {result['latency_ms']:.1f}ms")
        print(f"   速度: {result['speed_mbps']:.2f} MB/s")
        print(f"   URL: {result['base_url']}")
    else:
        print(f"\n❌ {result['name']}")
        print(f"   错误: {result.get('error', 'Unknown')}")

# 选择最佳镜像
best = find_best_mirror(results)

if best:
    print("\n" + "=" * 70)
    print("🏆 推荐使用以下镜像：")
    print("=" * 70)
    print(f"名称: {best['name']}")
    print(f"速度: {best['speed_mbps']:.2f} MB/s")
    print(f"延迟: {best['latency_ms']:.1f}ms")
    print(f"URL: {best['base_url']}")
    print("=" * 70)
    
    # 设置环境变量
    if best['use_mirror']:
        print(f"\n✅ 已自动设置环境变量: HF_ENDPOINT={best['base_url']}")
        os.environ["HF_ENDPOINT"] = best['base_url']
    else:
        print(f"\n✅ 使用官方源（无需设置环境变量）")
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
    
    # 保存推荐配置
    config_file = Path(__file__).parent.parent / "best_hf_mirror.txt"
    with open(config_file, 'w') as f:
        f.write(f"# 最佳HF镜像配置\n")
        f.write(f"# 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"HF_ENDPOINT={best['base_url']}\n")
        f.write(f"# 速度: {best['speed_mbps']:.2f} MB/s\n")
        f.write(f"# 延迟: {best['latency_ms']:.1f}ms\n")
    
    print(f"\n配置已保存到: {config_file}")
    print("\n可以开始下载模型了！")
    
    sys.exit(0)
else:
    print("\n❌ 无法找到可用的下载源")
    sys.exit(1)
