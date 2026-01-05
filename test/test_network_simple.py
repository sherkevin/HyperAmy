#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的网络连接测试（不依赖其他模块）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm import create_client
from llm.config import DEFAULT_MODEL
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_network_with_retry(prompt: str, max_retries: int = 3):
    """测试网络连接（带重试）"""
    client = create_client(model_name=DEFAULT_MODEL, mode="normal")
    
    for attempt in range(max_retries):
        try:
            answer = client.get_answer(prompt, max_tokens=100, temperature=0.7, mode="normal")
            return True, answer
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"尝试 {attempt + 1}/{max_retries} 失败: {error_msg[:100]}")
            
            if attempt == max_retries - 1:
                return False, f"[错误: {e}]"
            
            # 指数退避
            wait_time = 2 ** attempt
            logger.info(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
    
    return False, "[未知错误]"


def main():
    print("=" * 70)
    print("HyperAmy 网络连接测试")
    print("=" * 70)
    
    # 测试1: 简单API调用
    print("\n[测试1] 简单API调用...")
    success, answer = test_network_with_retry("请用一句话回答：1+1等于几？", max_retries=3)
    
    if success:
        print(f"✅ API调用成功")
        print(f"   回答: {answer[:100]}")
    else:
        print(f"❌ API调用失败: {answer}")
        print("\n请检查：")
        print("  1. 网络连接是否正常")
        print("  2. API密钥是否正确（检查 .env 文件）")
        print("  3. API服务是否可用")
        return False
    
    # 测试2: 连续多次调用（测试稳定性）
    print("\n[测试2] 连续多次调用（测试稳定性）...")
    success_count = 0
    total_tests = 5
    
    for i in range(total_tests):
        test_prompt = f"请用一句话回答：{i+1}的平方是多少？"
        success, answer = test_network_with_retry(test_prompt, max_retries=3)
        
        if success:
            success_count += 1
            print(f"   ✅ 调用 {i+1}/{total_tests} 成功: {answer[:50]}")
        else:
            print(f"   ❌ 调用 {i+1}/{total_tests} 失败")
        
        # 短暂延迟
        time.sleep(0.5)
    
    success_rate = success_count / total_tests * 100
    print(f"\n   成功率: {success_count}/{total_tests} ({success_rate:.1f}%)")
    
    # 测试3: 模拟实际使用场景（带上下文的问题）
    print("\n[测试3] 模拟实际使用场景（带上下文的问题）...")
    context = "这是一个测试文档。文档讲述了人工智能的发展历史。"
    question = "这个文档说了什么？"
    prompt = f"""基于以下上下文文档回答问题。如果上下文中没有相关信息，请说明无法从提供的上下文中找到答案。

上下文：
{context}

问题：{question}

答案："""
    
    success, answer = test_network_with_retry(prompt, max_retries=5)
    
    if success:
        print(f"✅ 带上下文的调用成功")
        print(f"   回答: {answer[:100]}")
    else:
        print(f"❌ 带上下文的调用失败: {answer}")
    
    # 总结
    print("\n" + "=" * 70)
    if success_rate >= 80:
        print("✅ 网络连接稳定，可以运行全量重试")
        print("\n下一步：运行全量重试")
        print("  python src/retry_failed_questions.py \\")
        print("      --input results/experiment_full.json \\")
        print("      --output results/experiment_full_retried.json")
    elif success_rate >= 50:
        print("⚠️  网络连接不稳定，但可以尝试运行重试")
        print("   建议：在网络较好的时候运行")
    else:
        print("❌ 网络连接不稳定，建议检查网络或API配置")
        print("   请检查：")
        print("   1. 网络连接")
        print("   2. API密钥（.env 文件中的 API_KEY）")
        print("   3. API服务状态")
    print("=" * 70)
    
    return success_rate >= 50


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

