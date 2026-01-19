#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在服务器上快速下载Qwen2.5-7B-Instruct模型

使用多线程和HF镜像加速下载
"""
import os
import sys
from pathlib import Path

# 优先使用HF镜像（中国用户更快）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置缓存目录（服务器路径）
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
cache_dir.parent.mkdir(parents=True, exist_ok=True)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)

print("=" * 70)
print("快速下载 Qwen2.5-7B-Instruct 模型（服务器端）")
print("=" * 70)
print(f"模型名称: Qwen/Qwen2.5-7B-Instruct")
print(f"HF镜像: {os.environ.get('HF_ENDPOINT')}")
print(f"缓存目录: {cache_dir}")
print("=" * 70)

# 检查磁盘空间
try:
    import shutil
    total, used, free = shutil.disk_usage(cache_dir.parent)
    free_gb = free / (1024**3)
    print(f"\n磁盘空间: {free_gb:.1f} GB 可用")
    if free_gb < 20:
        print("⚠️  警告：空间可能不足")
except:
    pass

print("\n开始下载（使用多线程和镜像加速）...\n")

try:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoConfig
    import torch
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 配置下载参数（加速）
    download_kwargs = {
        "cache_dir": str(cache_dir),
        "resume_download": True,
        "local_files_only": False,
    }
    
    # 如果安装了hf-transfer，可以使用更快的下载
    try:
        import hf_transfer
        print("✅ 检测到 hf-transfer，将使用更快的下载方式")
        download_kwargs["hf_transfer"] = True
    except ImportError:
        print("💡 提示：安装 hf-transfer 可以获得更快的下载速度")
        print("   pip install hf-transfer")
    
    # 步骤1: 下载tokenizer和配置文件
    print("步骤1/2: 下载tokenizer和配置文件...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            resume_download=True
        )
        print("✅ Tokenizer下载完成")
    except Exception as e:
        print(f"⚠️  Tokenizer下载警告: {e}")
        print("   继续下载模型文件...")
    
    # 步骤2: 下载完整模型（使用snapshot_download支持断点续传和多文件）
    print("\n步骤2/2: 下载模型权重文件（约14GB）...")
    print("   这可能需要较长时间，请耐心等待...")
    print("   支持断点续传，可以随时中断后重新运行\n")
    
    snapshot_download(
        repo_id=model_name,
        **download_kwargs
    )
    
    print("\n✅ 所有文件下载完成！")
    
    # 验证下载
    print("\n验证下载结果...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=str(cache_dir))
    
    print("\n" + "=" * 70)
    print("模型信息:")
    print("=" * 70)
    print(f"模型名称: {model_name}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Vocab size: {config.vocab_size}")
    if hasattr(config, 'max_position_embeddings'):
        print(f"Max position embeddings: {config.max_position_embeddings}")
    print(f"模型位置: {cache_dir}")
    print("=" * 70)
    
    # 检查文件大小
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    if model_dir.exists():
        import subprocess
        result = subprocess.run(
            ["du", "-sh", str(model_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            size = result.stdout.split()[0]
            print(f"\n总大小: {size}")
    
    print("\n✅ 模型下载并验证完成！")
    print(f"\n可以在代码中使用:")
    print(f"  from transformers import AutoModel")
    print(f"  model = AutoModel.from_pretrained('{model_name}')")

except ImportError as e:
    print("❌ 错误：缺少必要的库")
    print("\n请安装：")
    print("  pip install transformers huggingface_hub")
    if "hf_transfer" in str(e):
        print("  pip install hf-transfer  # 可选，加速下载")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\n⚠️  下载被中断")
    print("   可以重新运行此脚本继续下载")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
