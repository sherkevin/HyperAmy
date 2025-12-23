"""
使用 CompletionClient 的示例

这个文件展示了如何使用封装好的 CompletionClient
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import create_client

# 配置信息
API_KEY = "sk-7870u-nMQ69cSLRmIAxt2A"
MODEL_NAME = "DeepSeek-V3.2"

def get_full_sequence_probs(query):
    """
    获取完整序列的概率信息（使用新的 CompletionClient）
    
    Args:
        query: 用户查询
    """
    # 创建客户端
    client = create_client(API_KEY, model_name=MODEL_NAME)
    
    # 调用 API 并获取结果
    result = client.complete(query)
    
    # 打印详细分析
    result.print_analysis()

if __name__ == "__main__":
    get_full_sequence_probs("中国的首都是哪里？")