"""
测试 Speed (Surprise) 类

测试 chunk 惊讶值计算功能
"""
from particle.speed import Speed


def test_basic_extraction():
    """测试基本惊讶值提取"""
    print("=" * 60)
    print("测试 1: 基本惊讶值提取")
    print("=" * 60)
    
    speed = Speed()
    
    chunk = "这是一个测试文本。"
    surprise_value = speed.extract(chunk)
    
    print(f"Chunk: {chunk}")
    print(f"Surprise Value: {surprise_value:.4f}")
    print(f"说明: 值越大表示越意外/重要")
    print()


def test_different_aggregations():
    """测试不同的聚合方式"""
    print("=" * 60)
    print("测试 2: 不同聚合方式")
    print("=" * 60)
    
    speed = Speed()
    chunk = "量子纠缠现象颠覆了我们对现实的理解！"
    
    aggregations = ["mean", "sum", "max", "geometric_mean"]
    
    print(f"Chunk: {chunk}")
    print("-" * 60)
    print(f"{'聚合方式':<20} | {'惊讶值':<15}")
    print("-" * 60)
    
    for agg in aggregations:
        surprise_value = speed.extract(chunk, aggregation=agg)
        print(f"{agg:<20} | {surprise_value:<15.4f}")
    
    print()


def test_comparison():
    """测试不同内容的惊讶值比较"""
    print("=" * 60)
    print("测试 3: 不同内容的惊讶值比较")
    print("=" * 60)
    
    speed = Speed()
    
    chunks = [
        "这是一个普通的句子。",
        "今天天气很好。",
        "量子纠缠现象颠覆了我们对现实的理解！",
        "人工智能正在改变世界。",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    print(f"{'Chunk':<50} | {'惊讶值':<15}")
    print("-" * 70)
    
    results = []
    for chunk in chunks:
        surprise_value = speed.extract(chunk, aggregation="mean")
        results.append((chunk, surprise_value))
        print(f"{chunk:<50} | {surprise_value:<15.4f}")
    
    print()
    print("说明: 惊讶值越高，表示该内容越意外/重要")
    print()


def test_short_vs_long():
    """测试短文本和长文本的惊讶值"""
    print("=" * 60)
    print("测试 4: 短文本 vs 长文本（mean 聚合对长度不敏感）")
    print("=" * 60)
    
    speed = Speed()
    
    short_chunk = "重要！"
    long_chunk = "这是一个非常重要的信息，包含了大量的关键内容和细节描述，需要仔细分析和理解。"
    
    short_surprise = speed.extract(short_chunk, aggregation="mean")
    long_surprise = speed.extract(long_chunk, aggregation="mean")
    
    short_surprise_sum = speed.extract(short_chunk, aggregation="sum")
    long_surprise_sum = speed.extract(long_chunk, aggregation="sum")
    
    print(f"短文本: {short_chunk}")
    print(f"  平均惊讶值 (mean): {short_surprise:.4f}")
    print(f"  总惊讶值 (sum): {short_surprise_sum:.4f}")
    print()
    
    print(f"长文本: {long_chunk[:50]}...")
    print(f"  平均惊讶值 (mean): {long_surprise:.4f}")
    print(f"  总惊讶值 (sum): {long_surprise_sum:.4f}")
    print()
    
    print("说明:")
    print("  - mean 聚合对长度不敏感，适合比较不同长度的 chunk")
    print("  - sum 聚合对长度敏感，长文本的总惊讶值通常更大")
    print()


def test_max_aggregation():
    """测试 max 聚合方式（关注最意外的 token）"""
    print("=" * 60)
    print("测试 5: Max 聚合（关注最意外的 token）")
    print("=" * 60)
    
    speed = Speed()
    
    chunk = "这是一个包含特殊词汇 XYZ123ABC 的文本。"
    
    mean_surprise = speed.extract(chunk, aggregation="mean")
    max_surprise = speed.extract(chunk, aggregation="max")
    
    print(f"Chunk: {chunk}")
    print(f"平均惊讶值 (mean): {mean_surprise:.4f}")
    print(f"最大惊讶值 (max): {max_surprise:.4f}")
    print()
    print("说明: max 聚合关注最意外的单个 token，适合识别异常词汇")
    print()


def test_detailed_analysis():
    """测试详细分析（查看 token 级别的概率）"""
    print("=" * 60)
    print("测试 6: 详细分析（查看 token 概率）")
    print("=" * 60)
    
    speed = Speed()
    chunk = "量子纠缠"
    
    # 获取详细结果
    result = speed.client.complete(
        query=chunk,
        echo=True,
        max_tokens=0,
        temperature=0.0,
        logprobs=1
    )
    
    print(f"Chunk: {chunk}")
    print("-" * 60)
    print(f"{'Token':<20} | {'Logprob':<15} | {'Probability':<15} | {'Surprisal':<15}")
    print("-" * 60)
    
    for token_info in result.prompt_tokens:
        logprob = token_info.logprob if token_info.logprob is not None else float('-inf')
        prob = token_info.probability if token_info.probability is not None else 0.0
        surprisal = -logprob if logprob != float('-inf') else float('inf')
        
        logprob_str = f"{logprob:.4f}" if logprob != float('-inf') else "N/A"
        prob_str = f"{prob:.4f}" if prob > 0 else "N/A"
        surprisal_str = f"{surprisal:.4f}" if surprisal != float('inf') else "N/A"
        
        print(f"{repr(token_info.token):<20} | {logprob_str:<15} | {prob_str:<15} | {surprisal_str:<15}")
    
    print()
    
    # 计算整体惊讶值
    surprise_value = speed.extract(chunk)
    print(f"整体平均惊讶值: {surprise_value:.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Speed (Surprise) 类测试")
    print("=" * 60 + "\n")
    
    try:
        test_basic_extraction()
        test_different_aggregations()
        test_comparison()
        test_short_vs_long()
        test_max_aggregation()
        test_detailed_analysis()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

