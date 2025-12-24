"""
测试 CompletionClient 的使用
"""
from llm import CompletionClient, create_client
from llm.config import DEFAULT_MODEL

def test_basic_usage():
    """测试基本使用（普通对话模式）"""
    print("=" * 60)
    print("测试 1: 基本使用（normal 模式）")
    print("=" * 60)
    
    # 使用环境变量中的配置，默认 normal 模式
    client = create_client(model_name=DEFAULT_MODEL)
    result = client.complete("中国的首都是哪里？")
    
    print(f"回答: {result.get_answer_text()}")
    print(f"使用的 token 数: {result.usage.get('total_tokens', 0)}")
    print()

def test_detailed_analysis():
    """测试详细分析（使用 specific 模式获取 token 概率）"""
    print("=" * 60)
    print("测试 2: 详细分析（specific 模式 - 打印 token 概率）")
    print("=" * 60)
    
    client = create_client(model_name=DEFAULT_MODEL)
    # 使用 specific 模式获取 token 概率信息
    result = client.complete("中国的首都是哪里？", mode="specific")
    result.print_analysis()
    print()

def test_custom_prompt():
    """测试自定义 prompt（normal 模式）"""
    print("=" * 60)
    print("测试 3: 自定义 prompt 模板（normal 模式）")
    print("=" * 60)
    
    client = create_client(model_name=DEFAULT_MODEL)
    custom_template = "问题：{query}\n回答："
    result = client.complete("什么是量子力学？", prompt_template=custom_template)
    
    print(f"回答: {result.get_answer_text()}")
    print()

def test_quick_answer():
    """测试快速获取回答（normal 模式）"""
    print("=" * 60)
    print("测试 4: 快速获取回答（normal 模式）")
    print("=" * 60)
    
    client = create_client(model_name=DEFAULT_MODEL)
    answer = client.get_answer("Python 是什么？")
    print(f"回答: {answer}")
    print()

def test_custom_parameters():
    """测试自定义参数（normal 模式）"""
    print("=" * 60)
    print("测试 5: 自定义参数（normal 模式）")
    print("=" * 60)
    
    client = create_client(model_name=DEFAULT_MODEL)
    result = client.complete(
        "解释一下机器学习",
        max_tokens=200,
        temperature=0.5
    )
    print(f"回答: {result.get_answer_text()}")
    print(f"使用的 token 数: {result.usage.get('total_tokens', 0)}")
    print()

if __name__ == "__main__":
    # 运行所有测试
    test_basic_usage()
    test_detailed_analysis()
    test_custom_prompt()
    test_quick_answer()
    test_custom_parameters()

