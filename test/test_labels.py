"""
测试 Particle 类

测试情感向量、记忆深度和温度计算功能
"""
import numpy as np
from particle.particle import Particle


def test_basic_extraction():
    """测试基本提取（不使用 specific 模式）"""
    print("=" * 60)
    print("测试 1: 基本提取（不使用 specific 模式）")
    print("=" * 60)
    
    labels = Particle()
    
    chunk = "I'm very happy!"
    result = labels.extract(chunk, use_specific=False)
    
    print(f"Chunk: {chunk}")
    print(f"Emotion Vector Shape: {result.emotion_vector.shape}")
    print(f"Emotion Vector (完整): {result.emotion_vector}")
    print(f"Memory Depth: {result.memory_depth:.4f}")
    print(f"Temperature: {result.temperature}")
    print(f"说明: memory_depth 越大表示记忆越深刻，temperature 为 None（未使用 specific 模式）")
    print()


def test_with_specific_mode():
    """测试使用 specific 模式（包含 temperature）"""
    print("=" * 60)
    print("测试 2: 使用 specific 模式（包含 temperature）")
    print("=" * 60)
    
    labels = Particle()
    
    chunk = "I'm very happy!"
    result = labels.extract(chunk, use_specific=True)
    
    print(f"Chunk: {chunk}")
    print(f"Emotion Vector Shape: {result.emotion_vector.shape}")
    print(f"Emotion Vector (完整): {result.emotion_vector}")
    print(f"Memory Depth: {result.memory_depth:.4f}")
    print(f"Temperature: {result.temperature:.4f}" if result.temperature is not None else "Temperature: None")
    print(f"说明: temperature 越大表示情绪波动越大")
    print()


def test_memory_depth_comparison():
    """测试不同内容的记忆深度比较"""
    print("=" * 60)
    print("测试 3: 不同内容的记忆深度比较")
    print("=" * 60)
    
    labels = Particle()
    
    chunks = [
        "The weather is great today.",
        "I'm very happy!",
        "I love you deeply, and this feeling brings me immense happiness and contentment.",
        "Quantum entanglement overturns our understanding of reality!",
        "This is an ordinary sentence with no particular emotion.",
    ]
    
    print(f"{'Chunk':<50} | {'Memory Depth':<15} | {'Max Emotion':<15}")
    print("-" * 85)
    
    results = []
    for chunk in chunks:
        result = labels.extract(chunk, use_specific=False)
        max_emotion_idx = np.argmax(result.emotion_vector)
        max_emotion_value = result.emotion_vector[max_emotion_idx]
        results.append((chunk, result))
        
        chunk_display = chunk[:47] + "..." if len(chunk) > 50 else chunk
        print(f"{chunk_display:<50} | {result.memory_depth:<15.4f} | {max_emotion_value:<15.4f}")
    
    print()
    print("完整 Emotion Vectors:")
    print("-" * 85)
    for i, (chunk, result) in enumerate(results):
        print(f"\nChunk {i+1}: {chunk}")
        print(f"Emotion Vector: {result.emotion_vector}")
    
    print()
    print("说明: memory_depth 越大表示记忆越深刻（情绪越纯且强度越高）")
    print()


def test_temperature_comparison():
    """测试不同内容的温度比较"""
    print("=" * 60)
    print("测试 4: 不同内容的温度比较（使用 specific 模式）")
    print("=" * 60)
    
    labels = Particle()
    
    chunks = [
        "I'm very happy!",
        "I'm both happy and sad, my feelings are complicated.",
        "The weather is great today—sunny with a gentle breeze.",
        "Quantum entanglement overturns our understanding of reality!",
    ]
    
    print(f"{'Chunk':<50} | {'Memory Depth':<15} | {'Temperature':<15}")
    print("-" * 85)
    
    results = []
    for chunk in chunks:
        result = labels.extract(chunk, use_specific=True)
        results.append((chunk, result))
        chunk_display = chunk[:47] + "..." if len(chunk) > 50 else chunk
        temp_str = f"{result.temperature:.4f}" if result.temperature is not None else "N/A"
        print(f"{chunk_display:<50} | {result.memory_depth:<15.4f} | {temp_str:<15}")
    
    print()
    print("完整 Emotion Vectors:")
    print("-" * 85)
    for i, (chunk, result) in enumerate(results):
        print(f"\nChunk {i+1}: {chunk}")
        print(f"Emotion Vector: {result.emotion_vector}")
    
    print()
    print("说明:")
    print("  - temperature 越大表示情绪波动越大（纯度低或困惑度高）")
    print("  - temperature 越小表示情绪越稳定（纯度高且困惑度低）")
    print()


def test_emotion_vector_details():
    """测试情感向量详细信息"""
    print("=" * 60)
    print("测试 5: 情感向量详细信息")
    print("=" * 60)
    
    labels = Particle()
    
    chunk = "I love you deeply, and this feeling brings me immense happiness and contentment."
    result = labels.extract(chunk, use_specific=False)
    
    # 找到前5个最大的情绪分量
    top_indices = np.argsort(result.emotion_vector)[::-1][:5]
    
    print(f"Chunk: {chunk}")
    print(f"Emotion Vector Shape: {result.emotion_vector.shape}")
    print(f"Vector Sum: {np.sum(result.emotion_vector):.4f}")
    print(f"Vector Norm: {np.linalg.norm(result.emotion_vector):.4f}")
    print(f"Purity: {labels._compute_purity(result.emotion_vector):.4f}")
    print()
    print("Top 5 Emotions:")
    print("-" * 60)
    print(f"{'Emotion':<20} | {'Value':<15} | {'Percentage':<15}")
    print("-" * 60)
    
    from particle.particle import EMOTIONS
    total = np.sum(result.emotion_vector)
    for idx in top_indices:
        emotion_name = EMOTIONS[idx]
        value = result.emotion_vector[idx]
        percentage = (value / total * 100) if total > 0 else 0
        print(f"{emotion_name:<20} | {value:<15.4f} | {percentage:<14.2f}%")
    
    print()


def test_purity_and_temperature_relationship():
    """测试纯度和温度的关系"""
    print("=" * 60)
    print("测试 6: 纯度和温度的关系")
    print("=" * 60)
    
    labels = Particle()
    
    chunks = [
        "I'm very happy!",  # Single emotion, should have high purity and low temperature
        "I'm both happy and sad, my feelings are very complicated.",  # Mixed emotions, should have low purity and high temperature
        "I love you deeply.",  # Single strong emotion, should have high purity and low temperature
    ]
    
    print(f"{'Chunk':<50} | {'Purity':<15} | {'Temperature':<15} | {'Memory Depth':<15}")
    print("-" * 100)
    
    results = []
    for chunk in chunks:
        result = labels.extract(chunk, use_specific=True)
        purity = labels._compute_purity(result.emotion_vector)
        results.append((chunk, result, purity))
        temp_str = f"{result.temperature:.4f}" if result.temperature is not None else "N/A"
        
        chunk_display = chunk[:47] + "..." if len(chunk) > 50 else chunk
        print(f"{chunk_display:<50} | {purity:<15.4f} | {temp_str:<15} | {result.memory_depth:<15.4f}")
    
    print()
    print("完整 Emotion Vectors:")
    print("-" * 100)
    for i, (chunk, result, purity) in enumerate(results):
        print(f"\nChunk {i+1}: {chunk}")
        print(f"Emotion Vector: {result.emotion_vector}")
    
    print()
    print("说明:")
    print("  - 纯度越高，温度应该越低（情绪越纯且稳定）")
    print("  - 纯度越低，温度应该越高（情绪波动大）")
    print()


def test_memory_depth_calculation():
    """详细展示 memory_depth 的计算过程"""
    print("=" * 60)
    print("测试 7: Memory Depth 计算过程详解")
    print("=" * 60)
    
    labels = Particle()
    
    chunk = "I'm very happy!"
    result = labels.extract(chunk, use_specific=False)
    
    # 手动计算各个步骤
    emotion_vector = result.emotion_vector
    vector_sum = np.sum(emotion_vector)
    max_component = np.max(emotion_vector)
    magnitude = np.linalg.norm(emotion_vector)
    
    # 计算纯度
    purity = max_component / vector_sum if vector_sum > 0 else 0.0
    
    # 归一化模长
    normalized_magnitude = np.tanh(magnitude / labels.magnitude_scale)
    
    # 记忆深度
    memory_depth = purity * normalized_magnitude
    
    print(f"Chunk: {chunk}")
    print()
    print("步骤 1: 提取 Emotion Vector（不归一化）")
    print(f"  - Vector Shape: {emotion_vector.shape}")
    print(f"  - Vector Sum: {vector_sum:.4f}")
    print(f"  - Max Component: {max_component:.4f}")
    print(f"  - Vector (前5维): {emotion_vector[:5]}")
    print()
    
    print("步骤 2: 计算纯度 (Purity)")
    print(f"  - 公式: purity = max(emotion_vector) / sum(emotion_vector)")
    print(f"  - 计算: {max_component:.4f} / {vector_sum:.4f} = {purity:.4f}")
    print(f"  - 说明: 纯度表示单个情绪分量占比，越大表示情绪越纯")
    print()
    
    print("步骤 3: 计算模长 (Magnitude)")
    print(f"  - 公式: magnitude = ||emotion_vector|| (L2 norm)")
    print(f"  - 计算: ||{emotion_vector[:5]}...|| = {magnitude:.4f}")
    print(f"  - 说明: 模长表示情绪强度，保留原始强度信息")
    print()
    
    print("步骤 4: 归一化模长 (Normalized Magnitude)")
    print(f"  - 公式: normalized_magnitude = tanh(magnitude / magnitude_scale)")
    print(f"  - magnitude_scale: {labels.magnitude_scale}")
    print(f"  - 计算: tanh({magnitude:.4f} / {labels.magnitude_scale}) = {normalized_magnitude:.4f}")
    print(f"  - 说明: 使用 tanh 将模长映射到 0~1 范围")
    print()
    
    print("步骤 5: 计算记忆深度 (Memory Depth)")
    print(f"  - 公式: memory_depth = purity × normalized_magnitude")
    print(f"  - 计算: {purity:.4f} × {normalized_magnitude:.4f} = {memory_depth:.4f}")
    print(f"  - 说明: 记忆深度结合了纯度和模长，越大表示记忆越深刻")
    print()
    
    print("验证:")
    print(f"  - 实际结果: {result.memory_depth:.4f}")
    print(f"  - 计算值: {memory_depth:.4f}")
    print(f"  - 匹配: {'✓' if abs(result.memory_depth - memory_depth) < 1e-6 else '✗'}")
    print()


def test_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试 8: 边界情况")
    print("=" * 60)
    
    labels = Particle()
    
    edge_chunks = [
        "",  # Empty string
        "a",  # Single character
        "This is a very long text. " * 10,  # Long text
    ]
    
    print(f"{'Chunk (前50字符)':<50} | {'Memory Depth':<15} | {'Temperature':<15}")
    print("-" * 85)
    
    for chunk in edge_chunks:
        try:
            result = labels.extract(chunk, use_specific=False)
            chunk_display = (chunk[:47] + "...") if len(chunk) > 50 else chunk
            if len(chunk) == 0:
                chunk_display = "(empty)"
            print(f"{chunk_display:<50} | {result.memory_depth:<15.4f} | {'N/A':<15}")
        except Exception as e:
            chunk_display = (chunk[:47] + "...") if len(chunk) > 50 else chunk
            if len(chunk) == 0:
                chunk_display = "(empty)"
            print(f"{chunk_display:<50} | Error: {str(e)[:30]}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Particle 类测试")
    print("=" * 60 + "\n")
    
    try:
        test_basic_extraction()
        test_with_specific_mode()
        test_memory_depth_comparison()
        test_temperature_comparison()
        test_emotion_vector_details()
        test_purity_and_temperature_relationship()
        test_memory_depth_calculation()
        test_edge_cases()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

