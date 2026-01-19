#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载Qwen2.5-7B-Instruct模型

自动选择最快的下载源（如果已有测试结果）
"""
import os
import sys
from pathlib import Path

# 尝试加载最佳镜像配置
config_file = Path(__file__).parent.parent / "best_hf_mirror.txt"
if config_file.exists():
    with open(config_file, 'r') as f:
        for line in f:
            if line.startswith("HF_ENDPOINT="):
                endpoint = line.split("=", 1)[1].strip()
                os.environ["HF_ENDPOINT"] = endpoint
                print(f"✅ 使用测试的最佳镜像: {endpoint}")

# 如果没有配置，使用默认镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"使用默认镜像: {os.environ['HF_ENDPOINT']}")

# 设置缓存目录
os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path.home() / ".cache" / "huggingface" / "hub")

print("=" * 70)
print("下载 Qwen2.5-7B-Instruct 模型")
print("=" * 70)
print(f"模型名称: Qwen/Qwen2.5-7B-Instruct")
print(f"HF镜像: {os.environ.get('HF_ENDPOINT', 'default')}")
print(f"缓存目录: {os.environ.get('HUGGINGFACE_HUB_CACHE', 'default')}")
print("=" * 70)

# 检查磁盘空间（需要至少20GB）
cache_dir = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
cache_dir.parent.mkdir(parents=True, exist_ok=True)

try:
    import shutil
    total, used, free = shutil.disk_usage(cache_dir.parent)
    free_gb = free / (1024**3)
    print(f"\n磁盘空间检查:")
    print(f"  可用空间: {free_gb:.1f} GB")
    if free_gb < 20:
        print(f"  ⚠️  警告：可用空间不足20GB，可能无法完成下载")
        print(f"     建议至少保留25GB空间")
        response = input("\n是否继续？(y/n): ")
        if response.lower() != 'y':
            print("已取消下载")
            sys.exit(0)
    else:
        print(f"  ✅ 空间充足")
except Exception as e:
    print(f"  ⚠️  无法检查磁盘空间: {e}")

print("\n⚠️  注意：模型大小约14GB，下载需要较长时间")
print("   如果中断，可以重新运行此脚本继续下载")
print("   下载进度会显示在下方\n")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import logging
    from huggingface_hub import snapshot_download
    
    # 设置日志级别
    logging.set_verbosity_info()
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"步骤1/2: 下载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
            resume_download=True
        )
        print("✅ Tokenizer下载完成")
    except Exception as e:
        print(f"❌ Tokenizer下载失败: {e}")
        raise
    
    print(f"\n步骤2/2: 下载模型文件（这可能需要较长时间，约14GB）...")
    print("   提示：模型会保存到缓存目录，后续使用时会自动加载")
    print("   下载进度：")
    
    try:
        # 使用snapshot_download可以更好地显示进度
        print(f"\n开始下载模型文件到: {cache_dir}")
        print("（这会下载所有模型文件，包括权重、配置文件等）\n")
        
        # 先下载配置文件等小文件
        snapshot_download(
            repo_id=model_name,
            cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
            resume_download=True,
            ignore_patterns=["*.safetensors", "*.bin"]  # 先不下载大文件
        )
        
        print("\n配置文件下载完成，开始下载模型权重...")
        
        # 下载完整模型（包括权重）
        snapshot_download(
            repo_id=model_name,
            cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
            resume_download=True
        )
        
        print("\n✅ 所有模型文件下载完成")
        
        # 验证下载（尝试加载模型配置）
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        print("\n" + "=" * 70)
        print("模型信息:")
        print("=" * 70)
        print(f"模型名称: {model_name}")
        print(f"参数量: ~7B")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Vocab size: {config.vocab_size}")
        if hasattr(config, 'max_position_embeddings'):
            print(f"Max position embeddings: {config.max_position_embeddings}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 模型下载失败: {e}")
        print("\n可能的原因：")
        print("  1. 网络连接问题（请检查网络或VPN）")
        print("  2. 磁盘空间不足（需要至少20GB可用空间）")
        print("  3. HF镜像配置问题")
        print("\n建议：")
        print("  1. 检查网络连接")
        print("  2. 运行 python scripts/test_download_speed.py 重新测试最快的镜像")
        print("  3. 如果多次失败，可以尝试手动下载或使用其他网络")
        raise

except ImportError as e:
    print("❌ 错误：缺少必要的库")
    print("\n请安装必要的库：")
    print("  pip install transformers huggingface_hub")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\n⚠️  下载被中断")
    print("   可以重新运行此脚本继续下载（已下载的文件会被保留）")
    print(f"   已下载的文件保存在: {cache_dir}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ 下载过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ 所有文件下载完成！")
print("\n模型文件位置：")
print(f"  {cache_dir}")
print("\n可以在代码中使用：")
print(f"  from transformers import AutoModel")
print(f"  model = AutoModel.from_pretrained('{model_name}')")
print("\n或使用emos项目：")
print(f"  from emos.src.model import ProbabilisticGBERTV4")
print(f"  model = ProbabilisticGBERTV4(model_name='{model_name}')")
