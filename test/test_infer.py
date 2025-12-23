"""
使用 CompletionClient 的示例

这个文件展示了如何使用封装好的 CompletionClient
"""
from llm import create_client
from llm.config import DEFAULT_MODEL

def get_full_sequence_probs(query):
    """
    获取完整序列的概率信息（使用新的 CompletionClient）
    
    Args:
        query: 用户查询
    """
    # 创建客户端（使用环境变量中的配置）
    client = create_client(model_name=DEFAULT_MODEL)
    
    # 调用 API 并获取结果
    result = client.complete(query)
    
    # 打印详细分析
    result.print_analysis()

if __name__ == "__main__":
    get_full_sequence_probs("中国的首都是哪里？")