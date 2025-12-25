import requests
import numpy as np
import json
import re
import sys
import os

from llm.config import API_KEY, API_URL_CHAT

CHAT_API_URL = API_URL_CHAT

# 定义详细的情绪列表（基于Plutchik情绪轮和常见情绪）
sentimentS = [
    # 基本情绪（Plutchik的8种基本情绪）
    "joy",           # 快乐
    "sadness",       # 悲伤
    "anger",         # 愤怒
    "fear",          # 恐惧
    "surprise",      # 惊讶
    "disgust",       # 厌恶
    "trust",         # 信任
    "anticipation",  # 期待
    
    # 扩展情绪
    "love",          # 爱
    "hate",          # 恨
    "anxiety",       # 焦虑
    "calm",          # 平静
    "excitement",    # 兴奋
    "disappointment", # 失望
    "pride",         # 骄傲
    "shame",         # 羞耻
    "guilt",         # 愧疚
    "relief",        # 解脱
    "hope",          # 希望
    "despair",       # 绝望
    "contentment",   # 满足
    "frustration",   # 沮丧
    "gratitude",     # 感激
    "resentment",    # 怨恨
    "loneliness",    # 孤独
    "nostalgia",     # 怀旧
    "envy",          # 嫉妒
    "contempt",      # 轻蔑
]

def extract_sentiment_vector(text, model="Qwen3-Next-80B-A3B-Instruct"):
    """
    从文本中提取情绪向量
    
    Args:
        text: 输入文本
        model: 使用的模型名称
    
    Returns:
        numpy array: 归一化后的情绪向量
    """
    sentiments_str = ", ".join(sentimentS)
    
    prompt = f"""You are an sentiment analysis expert. Analyze the sentimental content of the given text and assign intensity scores (0.0 to 1.0) for each sentiment.

sentiment List:
{sentiments_str}

Instructions:
1. Read the text carefully
2. For each sentiment, assign a score from 0.0 (not present) to 1.0 (extremely strong)
3. Be precise - only assign high scores to sentiments that are clearly present
4. Output ONLY a JSON object with sentiment names as keys and scores as values
5. Do not include any explanation or additional text

Output Format (JSON only):
{{
  "joy": 0.8,
  "sadness": 0.1,
  "anger": 0.0,
  ...
}}

Text to analyze:
"{text}"

Output the JSON object only:"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,  # 低温度保证一致性
        "max_tokens": 500
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(CHAT_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # 提取JSON（可能包含markdown代码块）
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = content
        
        try:
            sentiment_dict = json.loads(json_str)
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试提取数字
            print(f"Warning: Failed to parse JSON, attempting to extract values...")
            print(f"Raw output: {content}")
            sentiment_dict = {}
            for sentiment in sentimentS:
                pattern = f'"{sentiment}"\\s*:\\s*([0-9.]+)'
                match = re.search(pattern, content)
                if match:
                    sentiment_dict[sentiment] = float(match.group(1))
                else:
                    sentiment_dict[sentiment] = 0.0
        
        # 构建向量（按照sentimentS的顺序）
        vector = np.array([sentiment_dict.get(sentiment, 0.0) for sentiment in sentimentS])
        
        # L2归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector, sentiment_dict
    else:
        raise Exception(f"Chat API Error: {response.status_code} - {response.text}")

def cosine_similarity(vec1, vec2):
    """
    计算两个情绪向量的余弦相似度（点积）
    
    注意：由于情绪向量所有分量都是非负值（0-1），归一化后仍在正象限，
    因此相似度范围是 [0, 1]，而不是 [-1, 1]
    - 1.0: 完全相同（所有情绪分布完全一致）
    - 0.0: 完全不同（没有共同的情绪）
    
    Returns:
        float: 相似度值，范围 [0, 1]
    """
    return np.dot(vec1, vec2)

def test_sentiment_similarity(text1, text2, description=""):
    """
    测试两个文本的情绪相似度
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        description: 测试描述
    
    Returns:
        similarity: 相似度值
    """
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"{'='*60}")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    
    vec1, dict1 = extract_sentiment_vector(text1)
    vec2, dict2 = extract_sentiment_vector(text2)
    
    # 显示主要情绪
    top_sentiments1 = sorted(dict1.items(), key=lambda x: x[1], reverse=True)[:5]
    top_sentiments2 = sorted(dict2.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\nTop sentiments for Text 1:")
    for sentiment, score in top_sentiments1:
        if score > 0.01:
            print(f"  {sentiment}: {score:.3f}")
    
    print(f"\nTop sentiments for Text 2:")
    for sentiment, score in top_sentiments2:
        if score > 0.01:
            print(f"  {sentiment}: {score:.3f}")
    
    similarity = cosine_similarity(vec1, vec2)
    print(f"\nsentiment Vector Similarity (range [0,1]): {similarity:.4f}")
    
    return similarity

# ===== 测试用例 =====

if __name__ == "__main__":
    print("="*60)
    print("sentiment Vector Similarity Testing")
    print("="*60)
    print(f"Using {len(sentimentS)} sentiment dimensions")
    print("="*60)
    
    # Category 1: Same sentiments, different descriptions
    print("\n" + "="*60)
    print("Category 1: Same sentiments, Different Descriptions")
    print("(Should have HIGH similarity)")
    print("="*60)
    
    test1 = test_sentiment_similarity(
        "I'm thrilled about winning the competition!",
        "The sunset over the ocean was absolutely breathtaking.",
        "1.1 Extreme happiness - different contexts"
    )
    
    test2 = test_sentiment_similarity(
        "I'm devastated by the loss of my pet.",
        "The storm destroyed everything in its path.",
        "1.2 Extreme sadness - different contexts"
    )
    
    test3 = test_sentiment_similarity(
        "I'm really pleased with how the project turned out.",
        "The coffee this morning tasted perfect, making me feel content.",
        "1.3 Moderate happiness - different sources"
    )
    
    test4 = test_sentiment_similarity(
        "I'm terrified of giving the presentation tomorrow.",
        "Walking alone in that dark alley made me feel extremely anxious.",
        "1.4 Fear/Anxiety - different sources"
    )
    
    test5 = test_sentiment_similarity(
        "I'm absolutely furious about the unfair treatment I received.",
        "The constant noise from the construction site is driving me insane with rage.",
        "1.5 Anger - different triggers"
    )
    
    # Category 2: Opposite sentiments, similar descriptions
    print("\n" + "="*60)
    print("Category 2: Opposite sentiments, Similar Descriptions")
    print("(Should have LOW similarity)")
    print("="*60)
    
    control1 = test_sentiment_similarity(
        "I love this new restaurant.",
        "I hate this new restaurant.",
        "2.1 Love vs Hate - same object"
    )
    
    control2 = test_sentiment_similarity(
        "I'm so excited about the upcoming vacation!",
        "I'm so disappointed about the cancelled vacation.",
        "2.2 Excitement vs Disappointment - same context"
    )
    
    control3 = test_sentiment_similarity(
        "I feel hopeful about my future prospects.",
        "I feel hopeless about my future prospects.",
        "2.3 Hope vs Despair - same situation"
    )
    
    control4 = test_sentiment_similarity(
        "I'm proud of my performance in the competition.",
        "I'm ashamed of my performance in the competition.",
        "2.4 Pride vs Shame - same achievement"
    )
    
    control5 = test_sentiment_similarity(
        "I'm grateful for my teacher's guidance.",
        "I resent my teacher's guidance.",
        "2.5 Gratitude vs Resentment - same person"
    )
    
    # Category 3: Different sentiment types
    print("\n" + "="*60)
    print("Category 3: Different sentiment Types")
    print("(Should have LOW similarity)")
    print("="*60)
    
    diff1 = test_sentiment_similarity(
        "I'm overjoyed about the upcoming trip!",
        "I'm terrified about the upcoming trip!",
        "3.1 Happiness vs Fear"
    )
    
    diff2 = test_sentiment_similarity(
        "I'm deeply saddened by the news.",
        "I'm extremely angry about the news.",
        "3.2 Sadness vs Anger"
    )
    
    diff3 = test_sentiment_similarity(
        "I was amazed by what I saw.",
        "I was disgusted by what I saw.",
        "3.3 Surprise vs Disgust"
    )
    
    diff4 = test_sentiment_similarity(
        "I love going to new places.",
        "I fear going to new places.",
        "3.4 Love vs Fear"
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    
    avg_same_sentiment = (test1 + test2 + test3 + test4 + test5) / 5
    avg_opposite_sentiment = (control1 + control2 + control3 + control4 + control5) / 5
    avg_different_types = (diff1 + diff2 + diff3 + diff4) / 4
    
    print(f"\nCategory 1 (Same sentiments, different desc):")
    print(f"  Average similarity: {avg_same_sentiment:.4f}")
    print(f"  Test 1: {test1:.4f}")
    print(f"  Test 2: {test2:.4f}")
    print(f"  Test 3: {test3:.4f}")
    print(f"  Test 4: {test4:.4f}")
    print(f"  Test 5: {test5:.4f}")
    
    print(f"\nCategory 2 (Opposite sentiments, similar desc):")
    print(f"  Average similarity: {avg_opposite_sentiment:.4f}")
    print(f"  Control 1: {control1:.4f}")
    print(f"  Control 2: {control2:.4f}")
    print(f"  Control 3: {control3:.4f}")
    print(f"  Control 4: {control4:.4f}")
    print(f"  Control 5: {control5:.4f}")
    
    print(f"\nCategory 3 (Different sentiment types):")
    print(f"  Average similarity: {avg_different_types:.4f}")
    print(f"  Diff 1: {diff1:.4f}")
    print(f"  Diff 2: {diff2:.4f}")
    print(f"  Diff 3: {diff3:.4f}")
    print(f"  Diff 4: {diff4:.4f}")
    
    print(f"\n{'='*60}")
    print("Key Comparison")
    print(f"{'='*60}")
    print(f"Same sentiments vs Opposite sentiments:")
    print(f"  Difference: {avg_same_sentiment - avg_opposite_sentiment:+.4f}")
    print(f"  Improvement: {(avg_same_sentiment - avg_opposite_sentiment) / avg_opposite_sentiment * 100:+.2f}%")
    
    print(f"\nSame sentiments vs Different types:")
    print(f"  Difference: {avg_same_sentiment - avg_different_types:+.4f}")
    print(f"  Improvement: {(avg_same_sentiment - avg_different_types) / avg_different_types * 100:+.2f}%")
    
    # 详细分析
    print(f"\n{'='*60}")
    print("Detailed Analysis")
    print(f"{'='*60}")
    
    # 计算分离度
    separation_ratio = avg_same_sentiment / avg_opposite_sentiment if avg_opposite_sentiment > 0 else float('inf')
    separation_ratio2 = avg_same_sentiment / avg_different_types if avg_different_types > 0 else float('inf')
    
    print(f"\n1. 分类效果评估:")
    print(f"   相同情绪 vs 相反情绪分离比: {separation_ratio:.2f}x")
    print(f"   相同情绪 vs 不同类型分离比: {separation_ratio2:.2f}x")
    
    print(f"\n2. 各类别统计:")
    print(f"   Category 1 (相同情绪):")
    print(f"     - 平均值: {avg_same_sentiment:.4f}")
    print(f"     - 范围: {min(test1, test2, test3, test4, test5):.4f} - {max(test1, test2, test3, test4, test5):.4f}")
    print(f"     - 标准差: {np.std([test1, test2, test3, test4, test5]):.4f}")
    
    print(f"\n   Category 2 (相反情绪):")
    print(f"     - 平均值: {avg_opposite_sentiment:.4f}")
    print(f"     - 范围: {min(control1, control2, control3, control4, control5):.4f} - {max(control1, control2, control3, control4, control5):.4f}")
    print(f"     - 标准差: {np.std([control1, control2, control3, control4, control5]):.4f}")
    
    print(f"\n   Category 3 (不同类型):")
    print(f"     - 平均值: {avg_different_types:.4f}")
    print(f"     - 范围: {min(diff1, diff2, diff3, diff4):.4f} - {max(diff1, diff2, diff3, diff4):.4f}")
    print(f"     - 标准差: {np.std([diff1, diff2, diff3, diff4]):.4f}")
    
    # 判断分类质量
    print(f"\n3. 分类质量评估:")
    
    # 检查是否有重叠
    min_same = min(test1, test2, test3, test4, test5)
    max_opposite = max(control1, control2, control3, control4, control5)
    max_different = max(diff1, diff2, diff3, diff4)
    
    overlap_opposite = min_same < max_opposite
    overlap_different = min_same < max_different
    
    if not overlap_opposite and not overlap_different:
        print(f"   ✓ 优秀: 三个类别完全分离，无重叠")
        print(f"     - 相同情绪最低值 ({min_same:.4f}) > 相反情绪最高值 ({max_opposite:.4f})")
        print(f"     - 相同情绪最低值 ({min_same:.4f}) > 不同类型最高值 ({max_different:.4f})")
    elif not overlap_opposite:
        print(f"   ✓ 良好: 相同情绪与相反情绪完全分离")
        print(f"     - 相同情绪最低值 ({min_same:.4f}) > 相反情绪最高值 ({max_opposite:.4f})")
        print(f"   ⚠ 注意: 与不同类型有轻微重叠")
    else:
        print(f"   ⚠ 需要改进: 存在类别重叠")
        if overlap_opposite:
            print(f"     - 相同情绪与相反情绪有重叠")
        if overlap_different:
            print(f"     - 相同情绪与不同类型有重叠")
    
    # 计算准确率（假设阈值）
    threshold = (avg_same_sentiment + avg_opposite_sentiment) / 2
    print(f"\n4. 基于阈值的分类准确率 (阈值={threshold:.4f}):")
    
    correct_same = sum(1 for s in [test1, test2, test3, test4, test5] if s > threshold)
    correct_opposite = sum(1 for s in [control1, control2, control3, control4, control5] if s < threshold)
    correct_different = sum(1 for s in [diff1, diff2, diff3, diff4] if s < threshold)
    
    total = 5 + 5 + 4
    correct = correct_same + correct_opposite + correct_different
    accuracy = correct / total * 100
    
    print(f"   - 相同情绪正确分类: {correct_same}/5 ({correct_same/5*100:.1f}%)")
    print(f"   - 相反情绪正确分类: {correct_opposite}/5 ({correct_opposite/5*100:.1f}%)")
    print(f"   - 不同类型正确分类: {correct_different}/4 ({correct_different/4*100:.1f}%)")
    print(f"   - 总体准确率: {correct}/{total} ({accuracy:.1f}%)")
    
    if avg_same_sentiment > avg_opposite_sentiment and avg_same_sentiment > avg_different_types:
        print(f"\n{'='*60}")
        print("✓ SUCCESS: sentiment vector method successfully distinguishes sentiments!")
        print(f"{'='*60}")
        print(f"  - Same sentiments: {avg_same_sentiment:.4f} (high similarity)")
        print(f"  - Opposite sentiments: {avg_opposite_sentiment:.4f} (low similarity)")
        print(f"  - Different types: {avg_different_types:.4f} (low similarity)")
        print(f"\n  结论: 该方法能够有效区分情绪相似度，")
        print(f"        相同情绪文本具有高相似度，相反情绪文本具有低相似度。")
    else:
        print(f"\n{'='*60}")
        print("⚠ Results need improvement:")
        print(f"{'='*60}")
        print(f"  - Same sentiments: {avg_same_sentiment:.4f}")
        print(f"  - Opposite sentiments: {avg_opposite_sentiment:.4f}")
        print(f"  - Different types: {avg_different_types:.4f}")

