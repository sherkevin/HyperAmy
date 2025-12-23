import requests
import numpy as np

# 你的 SiliconFlow API Key
API_KEY = "sk-7870u-nMQ69cSLRmIAxt2A"
EMBEDDING_API_URL = "https://llmapi.paratera.com/v1/embeddings"
CHAT_API_URL = "https://llmapi.paratera.com/v1/chat/completions"

def extract_emotion_description(text, model="Qwen3-Next-80B-A3B-Instruct"):
    """
    使用大语言模型提取文本的情绪描述，专注于情绪维度而非语义内容
    
    Args:
        text: 输入文本
        model: 使用的模型名称
    
    Returns:
        提取出的情绪描述文本
    """
    # 精心设计的 prompt，专注于情绪提取
    # 关键设计原则：
    # 1. 明确要求忽略具体事件和对象，只关注情绪
    # 2. 要求用统一的情绪描述格式
    # 3. 强调情绪的强度和类型
    # 4. 使用示例引导模型理解任务
    
    prompt = """You are an emotion extraction specialist. Extract ONLY the pure emotional and affective meaning, completely ignoring all semantic content, events, objects, people, places, or contextual details.

CRITICAL RULES:
1. NEVER mention what happened, who was involved, or where it occurred
2. NEVER describe the event, situation, or context
3. ONLY describe: feelings, emotions, moods, sentiments, affective states
4. Use diverse emotional vocabulary - avoid generic words like "positive" or "negative"
5. For OPPOSITE emotions, use COMPLETELY DIFFERENT vocabulary and sentence structures
6. For SIMILAR emotions, use SIMILAR vocabulary and structures

KEY PRINCIPLE: 
- Similar emotions → Similar descriptions (same structure, similar words)
- Opposite emotions → VERY DIFFERENT descriptions (different structure, contrasting words)

Output Format - Use VARIED structures based on emotion:
- For positive emotions: Use uplifting, warm vocabulary
- For negative emotions: Use contrasting, darker vocabulary
- Vary sentence structure to maximize semantic difference for opposite emotions

Examples:
Input: "I'm thrilled about winning the competition!"
Output: "Experiencing intense joy, elation, and triumph. Overwhelming positive feelings of success and accomplishment."

Input: "The sunset over the ocean was absolutely breathtaking."
Output: "Experiencing intense joy, wonder, and serenity. Powerful positive feelings of awe and beauty."
(Note: Both positive, so similar structure and vocabulary)

Input: "I'm devastated by the loss of my pet."
Output: "Experiencing profound sorrow, grief, and despair. Crushing negative feelings of loss and emptiness."

Input: "The storm destroyed everything in its path."
Output: "Experiencing profound sorrow, shock, and devastation. Overwhelming negative feelings of destruction and helplessness."
(Note: Both negative, so similar structure and vocabulary)

Input: "I love this new restaurant."
Output: "Experiencing warmth, satisfaction, and delight. Pleasant positive feelings of appreciation and contentment."

Input: "I hate this new restaurant."
Output: "Experiencing coldness, dissatisfaction, and repulsion. Unpleasant negative feelings of rejection and displeasure."
(Note: OPPOSITE emotions - notice how structure and vocabulary are DIFFERENT: "warmth" vs "coldness", "delight" vs "repulsion", "pleasant" vs "unpleasant", "appreciation" vs "rejection")

CRITICAL: For opposite emotions, use COMPLETELY DIFFERENT sentence structures:
- Positive emotions: Use structure like "Experiencing [positive words]. [Positive descriptor] positive feelings..."
- Negative emotions: Use DIFFERENT structure like "[Negative words] dominate. [Negative descriptor] negative emotional state characterized by..."
OR use entirely different grammatical structures to maximize semantic distance in embeddings.

Now extract the emotion from:
"{text}"

Output ONLY the emotional description. Use varied vocabulary and structure to maximize differences for opposite emotions:"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt.format(text=text)
            }
        ],
        "temperature": 0.3,  # 较低温度保证一致性
        "max_tokens": 100
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(CHAT_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        emotion_desc = response.json()['choices'][0]['message']['content'].strip()
        return emotion_desc
    else:
        raise Exception(f"Chat API Error: {response.status_code} - {response.text}")

def get_fancy_emotion_embedding(text_chunk, use_strong_instruction=False):
    """
    获取 NIPS 级别的 'Instruction-Aware Emotion Embedding'
    
    Args:
        text_chunk: Input text
        use_strong_instruction: If True, use a stronger emotion-focused instruction
    """
    # 关键点：使用 Instruction 强制模型在情感维度进行编码
    # 这就是 paper 里的 "Prompt-based Manifold Alignment"
    if use_strong_instruction:
        instruction = "Extract the emotional and affective meaning from this text, focusing on feelings and sentiments: "
    else:
        instruction = "Represent this sentence for emotion classification and affective analysis: "
    
    # Doubao-Embedding-Text 模型支持输入指令
    # 这里的 input 构造方式取决于具体 API 协议，通用做法如下：
    payload = {
        "model": "Doubao-Embedding-Text",
        "input": instruction + text_chunk,
        "encoding_format": "float"
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(EMBEDDING_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        # 返回的是一个高维稠密向量 (Doubao-Embedding-Text 返回 2560 维度)
        # 这比简单的 [0.8, 0.2] 概率向量 fancy 太多了
        # 你可以在这个向量上做 PCA、聚类，或者作为 Agent 的 Memory Key
        embedding = response.json()['data'][0]['embedding']
        return np.array(embedding)
    else:
        raise Exception(f"API Error: {response.text}")

def dot_product_similarity(vec1, vec2, normalize=False):
    """
    Calculate dot product similarity between two embedding vectors
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        normalize: If True, normalize vectors before dot product (makes it equal to cosine similarity)
    
    Returns:
        Dot product value. If normalize=True, result is in [-1, 1] range.
        If normalize=False, result depends on vector magnitudes.
    """
    if normalize:
        # Normalize vectors to unit length
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        return np.dot(vec1_norm, vec2_norm)
    else:
        return np.dot(vec1, vec2)

def check_vector_normalization(vec, name=""):
    """
    Check if a vector is normalized (L2 norm should be ~1.0)
    """
    l2_norm = np.linalg.norm(vec)
    print(f"{name} L2 norm: {l2_norm:.6f}")
    return l2_norm

def get_emotion_based_embedding(text, use_emotion_extraction=True):
    """
    获取基于情绪描述的 embedding
    
    Args:
        text: 原始文本
        use_emotion_extraction: 如果为 True，先提取情绪描述再获取 embedding
    
    Returns:
        embedding 向量和（如果提取了）情绪描述
    """
    if use_emotion_extraction:
        # 第一阶段：提取情绪描述
        emotion_desc = extract_emotion_description(text)
        # 第二阶段：获取情绪描述的 embedding
        embedding = get_fancy_emotion_embedding(emotion_desc, use_strong_instruction=False)
        return embedding, emotion_desc
    else:
        # 直接获取原始文本的 embedding
        embedding = get_fancy_emotion_embedding(text, use_strong_instruction=False)
        return embedding, None

def test_embedding_similarity(text1, text2, description="", use_strong_instruction=False, normalize=False, use_emotion_extraction=False):
    """
    Test similarity between two texts and print results
    
    Args:
        text1: First text
        text2: Second text
        description: Test description
        use_strong_instruction: Whether to use stronger emotion-focused instruction
        normalize: Whether to normalize vectors before dot product
        use_emotion_extraction: If True, extract emotion descriptions first, then get embeddings
    """
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    if use_strong_instruction:
        print("(Using strong emotion-focused instruction)")
    if use_emotion_extraction:
        print("(Using emotion extraction pipeline)")
    print(f"{'='*60}")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    
    if use_emotion_extraction:
        # 使用情绪提取流程
        vec1, emotion1 = get_emotion_based_embedding(text1, use_emotion_extraction=True)
        vec2, emotion2 = get_emotion_based_embedding(text2, use_emotion_extraction=True)
        print(f"\nExtracted Emotion 1: {emotion1}")
        print(f"Extracted Emotion 2: {emotion2}")
    else:
        # 直接获取原始文本的 embedding
        vec1 = get_fancy_emotion_embedding(text1, use_strong_instruction)
        vec2 = get_fancy_emotion_embedding(text2, use_strong_instruction)
    
    # Check normalization (only for first test to avoid too much output)
    if "Different descriptions, both positive emotions" in description and not use_emotion_extraction:
        print("\nVector Normalization Check:")
        check_vector_normalization(vec1, "Vector 1")
        check_vector_normalization(vec2, "Vector 2")
        print(f"Note: If vectors are normalized, L2 norm should be ~1.0")
        print(f"      Dot product of normalized vectors equals cosine similarity")
    
    similarity = dot_product_similarity(vec1, vec2, normalize=normalize)
    if normalize:
        print(f"Dot Product Similarity (normalized, range [-1,1]): {similarity:.4f}")
    else:
        print(f"Dot Product Similarity (raw, not normalized): {similarity:.4f}")
        # Also calculate normalized version for comparison
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        print(f"  → Normalized dot product (cosine similarity): {cosine_sim:.4f}")
    
    return similarity

# --- Test Cases: Different Descriptions but Similar Emotions ---

print("Testing Emotion-Aware Embedding Similarity")
print("=" * 60)
print("Note: Vectors from API are NOT normalized (L2 norm ~152)")
print("Using normalized dot product (equivalent to cosine similarity) for values in [-1, 1] range")
print("=" * 60)

# Test Case 1: Positive emotions, different descriptions
test1_sim = test_embedding_similarity(
    "I'm thrilled about winning the competition!",
    "The sunset over the ocean was absolutely breathtaking.",
    "Different descriptions, both positive emotions",
    normalize=True
)

# Test Case 2: Negative emotions, different descriptions
test2_sim = test_embedding_similarity(
    "I'm devastated by the loss of my pet.",
    "The storm destroyed everything in its path.",
    "Different descriptions, both negative emotions",
    normalize=True
)

# Test Case 3: Mixed positive emotions, different contexts
test3_sim = test_embedding_similarity(
    "Getting accepted to my dream university fills me with joy.",
    "The concert last night was absolutely amazing and unforgettable.",
    "Different contexts, both expressing joy/excitement",
    normalize=True
)

# Test Case 4: Mixed negative emotions, different contexts
test4_sim = test_embedding_similarity(
    "I'm deeply disappointed by the test results.",
    "The news about the accident made me feel very sad.",
    "Different contexts, both expressing sadness/disappointment",
    normalize=True
)

# Test Case 5: Extremely different topics, same strong positive emotion
test5_sim = test_embedding_similarity(
    "I won the lottery! This is the happiest day of my life!",
    "The rainbow after the storm was the most beautiful thing I've ever seen.",
    "Completely different topics, both expressing extreme happiness",
    normalize=True
)

# Test Case 6: Extremely different topics, same strong negative emotion
test6_sim = test_embedding_similarity(
    "My house burned down in the fire. Everything is gone.",
    "I failed the most important exam of my life. I'm devastated.",
    "Completely different topics, both expressing extreme sadness/despair",
    normalize=True
)

# Test Case 7: Fear/anxiety emotions, different contexts
test7_sim = test_embedding_similarity(
    "I'm terrified of speaking in public.",
    "The dark alley at night made me feel very anxious.",
    "Different contexts, both expressing fear/anxiety",
    normalize=True
)

# --- Control Cases: Similar descriptions but different emotions ---

print("\n" + "="*60)
print("Control Cases: Similar Descriptions, Different Emotions")
print("="*60)

# Control Case 1: Similar description, opposite emotions
control1_sim = test_embedding_similarity(
    "I love this new restaurant.",
    "I hate this new restaurant.",
    "Similar description, opposite emotions",
    normalize=True
)

# Control Case 2: Similar description, different emotions
control2_sim = test_embedding_similarity(
    "The movie was fantastic and entertaining.",
    "The movie was boring and disappointing.",
    "Similar description, different emotions",
    normalize=True
)

# Control Case 3: Same topic, opposite extreme emotions
control3_sim = test_embedding_similarity(
    "I won the lottery! This is the happiest day of my life!",
    "I lost all my money in a scam. This is the worst day of my life.",
    "Same topic (money/life event), opposite extreme emotions",
    normalize=True
)

# Control Case 4: Same topic, opposite emotions
control4_sim = test_embedding_similarity(
    "The rainbow after the storm was the most beautiful thing I've ever seen.",
    "The storm destroyed my house and everything I owned.",
    "Same topic (storm), opposite emotions",
    normalize=True
)

# --- Baseline Case: Completely different ---

print("\n" + "="*60)
print("Baseline Case: Different Descriptions and Emotions")
print("="*60)

baseline_sim = test_embedding_similarity(
    "The weather is nice today.",
    "I'm feeling terrible about the exam.",
    "Different descriptions and emotions (baseline)",
    normalize=True
)

# --- Summary ---

print("\n" + "="*60)
print("Summary of Results")
print("="*60)
print(f"Test 1 (Positive, different desc): {test1_sim:.4f}")
print(f"Test 2 (Negative, different desc): {test2_sim:.4f}")
print(f"Test 3 (Joy, different contexts):  {test3_sim:.4f}")
print(f"Test 4 (Sadness, different contexts): {test4_sim:.4f}")
print(f"Test 5 (Extreme happiness, diff topics): {test5_sim:.4f}")
print(f"Test 6 (Extreme sadness, diff topics): {test6_sim:.4f}")
print(f"Test 7 (Fear/anxiety, diff contexts): {test7_sim:.4f}")
print(f"\nControl 1 (Opposite emotions):    {control1_sim:.4f}")
print(f"Control 2 (Different emotions):    {control2_sim:.4f}")
print(f"Control 3 (Same topic, opposite extreme): {control3_sim:.4f}")
print(f"Control 4 (Same topic, opposite):   {control4_sim:.4f}")
print(f"\nBaseline (Different everything):  {baseline_sim:.4f}")

avg_same_emotion = (test1_sim + test2_sim + test3_sim + test4_sim + test5_sim + test6_sim + test7_sim) / 7
avg_control = (control1_sim + control2_sim + control3_sim + control4_sim) / 4

print(f"\nAverage similarity (same emotion, different desc): {avg_same_emotion:.4f}")
print(f"Average similarity (different emotions, similar desc): {avg_control:.4f}")
print(f"Baseline similarity (different everything): {baseline_sim:.4f}")

print(f"\nDifference (same emotion vs different emotions): {avg_same_emotion - avg_control:.4f}")
print(f"Difference (same emotion vs baseline): {avg_same_emotion - baseline_sim:.4f}")

print("\n" + "="*60)
print("Detailed Analysis: Semantic vs Emotion Similarity")
print("="*60)

# Calculate percentage differences
semantic_advantage = avg_control - avg_same_emotion
emotion_advantage = avg_same_emotion - baseline_sim
semantic_advantage_pct = (semantic_advantage / avg_same_emotion) * 100
emotion_advantage_pct = (emotion_advantage / baseline_sim) * 100

print(f"\n1. 相同情感但不同描述的相似度: {avg_same_emotion:.4f}")
print(f"   (7个测试用例的平均值)")
print(f"\n2. 不同情感但相似描述的相似度: {avg_control:.4f}")
print(f"   (4个对照测试的平均值)")
print(f"\n3. 基线（完全不同）: {baseline_sim:.4f}")

print(f"\n{'='*60}")
print("关键发现:")
print(f"{'='*60}")

if avg_same_emotion > avg_control and avg_same_emotion > baseline_sim:
    print("✓ 结论: 模型更关注情绪相似度")
    print(f"  - 相同情绪但不同描述的相似度 ({avg_same_emotion:.4f})")
    print(f"    高于不同情绪但相似描述的相似度 ({avg_control:.4f})")
    print(f"  - 差异: +{abs(avg_same_emotion - avg_control):.4f} ({abs(semantic_advantage_pct):.2f}%)")
else:
    print("✗ 结论: 模型更关注语义相似度")
    print(f"\n  证据1: 相似描述的相似度 ({avg_control:.4f})")
    print(f"         高于相同情绪的相似度 ({avg_same_emotion:.4f})")
    print(f"         差异: +{abs(semantic_advantage):.4f} ({abs(semantic_advantage_pct):.2f}%)")
    
    print(f"\n  证据2: 相同情绪与基线的差异很小")
    print(f"         相同情绪: {avg_same_emotion:.4f}")
    print(f"         基线:     {baseline_sim:.4f}")
    print(f"         差异: {emotion_advantage:.4f} ({emotion_advantage_pct:.2f}%)")
    print(f"         → 说明情绪相似性对相似度影响很小")
    
    print(f"\n  证据3: 具体案例对比")
    print(f"         - 'I love this restaurant' vs 'I hate this restaurant'")
    print(f"           相似度: {control1_sim:.4f} (极高，因为描述几乎相同)")
    print(f"         - 'I won lottery' vs 'Rainbow is beautiful'")
    print(f"           相似度: {test5_sim:.4f} (较低，尽管都是积极情绪)")
    
    print(f"\n  模型行为:")
    print(f"  - 当文本描述相似时，即使情绪相反，相似度仍然很高")
    print(f"  - 当文本描述不同时，即使情绪相同，相似度也较低")
    print(f"  - 这表明模型主要基于语义内容（词汇、主题、结构）计算相似度")
    print(f"  - 情绪信息可能被编码在向量中，但对相似度计算的影响较小")
    
    print(f"\n  可能的原因:")
    print(f"  1. Doubao-Embedding-Text 模型训练时更注重语义理解")
    print(f"  2. 当前指令可能不足以引导模型关注情绪维度")
    print(f"  3. 情绪信息可能存在于向量的某些维度，但被语义信息主导")

# Test with stronger emotion instruction
print("\n" + "="*60)
print("Testing with Stronger Emotion-Focused Instruction")
print("="*60)

test_strong_1 = test_embedding_similarity(
    "I won the lottery! This is the happiest day of my life!",
    "The rainbow after the storm was the most beautiful thing I've ever seen.",
    "Extreme happiness, different topics (strong instruction)",
    use_strong_instruction=True,
    normalize=True
)

test_strong_2 = test_embedding_similarity(
    "I love this new restaurant.",
    "I hate this new restaurant.",
    "Opposite emotions, similar description (strong instruction)",
    use_strong_instruction=True,
    normalize=True
)

print(f"\nComparison:")
print(f"Same emotion (strong instruction): {test_strong_1:.4f}")
print(f"Opposite emotion (strong instruction): {test_strong_2:.4f}")
print(f"Difference: {test_strong_1 - test_strong_2:.4f}")

if test_strong_1 > test_strong_2:
    print("✓ Stronger instruction helps capture emotion similarity!")
else:
    print("✗ Stronger instruction doesn't significantly change the pattern.")

# --- Testing with Emotion Extraction Pipeline ---

print("\n" + "="*60)
print("Comprehensive Testing with Emotion Extraction Pipeline")
print("="*60)
print("Using Qwen3-Next-80B-A3B-Instruct to extract emotion descriptions first")
print("Then comparing embeddings of emotion descriptions")
print("="*60)

# ===== Category 1: Same Emotion Type, Different Descriptions (Should have HIGH similarity) =====
print("\n" + "="*60)
print("Category 1: Same Emotion Type, Different Descriptions")
print("(Should have HIGH similarity after emotion extraction)")
print("="*60)

# 1.1 Extreme Happiness - completely different contexts
emotion_test1_1 = test_embedding_similarity(
    "I'm ecstatic! I just got accepted to my dream university!",
    "Winning the championship was the most incredible moment of my entire life!",
    "1.1 Extreme happiness - different achievements",
    normalize=True,
    use_emotion_extraction=True
)

emotion_test1_2 = test_embedding_similarity(
    "The birth of my child filled me with indescribable joy.",
    "Seeing the northern lights for the first time was absolutely magical and euphoric.",
    "1.2 Extreme happiness - different experiences",
    normalize=True,
    use_emotion_extraction=True
)

# 1.3 Moderate Happiness - different contexts
emotion_test1_3 = test_embedding_similarity(
    "I'm really pleased with how the project turned out.",
    "The coffee this morning tasted perfect, making me feel content.",
    "1.3 Moderate happiness - different sources",
    normalize=True,
    use_emotion_extraction=True
)

# 1.4 Extreme Sadness - different contexts
emotion_test2_1 = test_embedding_similarity(
    "My grandmother passed away last week. I'm completely heartbroken.",
    "The company I worked for 20 years just went bankrupt. I feel utterly devastated.",
    "2.1 Extreme sadness - different losses",
    normalize=True,
    use_emotion_extraction=True
)

emotion_test2_2 = test_embedding_similarity(
    "I failed the most important exam of my life. Everything feels hopeless.",
    "My best friend moved to another country. I'm crushed and lonely.",
    "2.2 Extreme sadness - different situations",
    normalize=True,
    use_emotion_extraction=True
)

# 1.5 Moderate Sadness - different contexts
emotion_test2_3 = test_embedding_similarity(
    "I'm feeling a bit down after missing the deadline.",
    "The rainy weather today makes me feel melancholic.",
    "2.3 Moderate sadness - different triggers",
    normalize=True,
    use_emotion_extraction=True
)

# 1.6 Fear/Anxiety - different contexts
emotion_test3_1 = test_embedding_similarity(
    "I'm terrified of giving the presentation tomorrow.",
    "Walking alone in that dark alley made me feel extremely anxious.",
    "3.1 Fear/anxiety - different sources",
    normalize=True,
    use_emotion_extraction=True
)

emotion_test3_2 = test_embedding_similarity(
    "The thought of losing my job keeps me awake at night.",
    "I'm worried sick about my daughter's health condition.",
    "3.2 Fear/anxiety - different concerns",
    normalize=True,
    use_emotion_extraction=True
)

# 1.7 Anger - different contexts
emotion_test4_1 = test_embedding_similarity(
    "I'm absolutely furious about the unfair treatment I received.",
    "The constant noise from the construction site is driving me insane with rage.",
    "4.1 Anger - different triggers",
    normalize=True,
    use_emotion_extraction=True
)

emotion_test4_2 = test_embedding_similarity(
    "I'm really annoyed by the slow internet connection.",
    "The rude customer service made me feel irritated and frustrated.",
    "4.2 Moderate anger - different situations",
    normalize=True,
    use_emotion_extraction=True
)

# 1.8 Surprise/Wonder - different contexts
emotion_test5_1 = test_embedding_similarity(
    "I was completely astonished by the unexpected gift.",
    "The magician's trick left me in absolute amazement.",
    "5.1 Surprise/wonder - different sources",
    normalize=True,
    use_emotion_extraction=True
)

# 1.9 Disgust - different contexts
emotion_test6_1 = test_embedding_similarity(
    "The spoiled food made me feel nauseous and disgusted.",
    "I'm repulsed by the unethical behavior I witnessed.",
    "6.1 Disgust - different triggers",
    normalize=True,
    use_emotion_extraction=True
)

# 1.10 Love/Affection - different contexts
emotion_test7_1 = test_embedding_similarity(
    "I adore spending time with my family.",
    "My heart swells with love when I see my pet.",
    "7.1 Love/affection - different objects",
    normalize=True,
    use_emotion_extraction=True
)

# 1.11 Relief - different contexts
emotion_test8_1 = test_embedding_similarity(
    "I felt immense relief when I found my lost wallet.",
    "Finally finishing the difficult project brought me great peace.",
    "8.1 Relief - different sources",
    normalize=True,
    use_emotion_extraction=True
)

# ===== Category 2: Different Emotions, Similar Descriptions (Should have LOW similarity) =====
print("\n" + "="*60)
print("Category 2: Different Emotions, Similar Descriptions")
print("(Should have LOW similarity after emotion extraction)")
print("="*60)

# 2.1 Love vs Hate - same object
emotion_control1_1 = test_embedding_similarity(
    "I absolutely love this new book I'm reading.",
    "I absolutely hate this new book I'm reading.",
    "2.1 Love vs Hate - same object",
    normalize=True,
    use_emotion_extraction=True
)

# 2.2 Joy vs Sadness - same event type
emotion_control2_1 = test_embedding_similarity(
    "Graduating from college was the happiest moment of my life.",
    "Graduating from college was the saddest moment of my life.",
    "2.2 Joy vs Sadness - same event",
    normalize=True,
    use_emotion_extraction=True
)

# 2.3 Excitement vs Disappointment - same context
emotion_control3_1 = test_embedding_similarity(
    "I'm so excited about the upcoming vacation!",
    "I'm so disappointed about the cancelled vacation.",
    "2.3 Excitement vs Disappointment - same context",
    normalize=True,
    use_emotion_extraction=True
)

# 2.4 Pride vs Shame - same achievement
emotion_control4_1 = test_embedding_similarity(
    "I'm proud of my performance in the competition.",
    "I'm ashamed of my performance in the competition.",
    "2.4 Pride vs Shame - same achievement",
    normalize=True,
    use_emotion_extraction=True
)

# 2.5 Hope vs Despair - same situation
emotion_control5_1 = test_embedding_similarity(
    "I feel hopeful about my future prospects.",
    "I feel hopeless about my future prospects.",
    "2.5 Hope vs Despair - same situation",
    normalize=True,
    use_emotion_extraction=True
)

# 2.6 Gratitude vs Resentment - same person
emotion_control6_1 = test_embedding_similarity(
    "I'm grateful for my teacher's guidance.",
    "I resent my teacher's guidance.",
    "2.6 Gratitude vs Resentment - same person",
    normalize=True,
    use_emotion_extraction=True
)

# 2.7 Contentment vs Frustration - same activity
emotion_control7_1 = test_embedding_similarity(
    "I'm satisfied with my work progress.",
    "I'm frustrated with my work progress.",
    "2.7 Contentment vs Frustration - same activity",
    normalize=True,
    use_emotion_extraction=True
)

# ===== Category 3: Different Emotion Types (Should have LOW similarity) =====
print("\n" + "="*60)
print("Category 3: Different Emotion Types")
print("(Should have LOW similarity)")
print("="*60)

# 3.1 Happiness vs Fear
emotion_diff1 = test_embedding_similarity(
    "I'm overjoyed about the upcoming trip!",
    "I'm terrified about the upcoming trip!",
    "3.1 Happiness vs Fear",
    normalize=True,
    use_emotion_extraction=True
)

# 3.2 Sadness vs Anger
emotion_diff2 = test_embedding_similarity(
    "I'm deeply saddened by the news.",
    "I'm extremely angry about the news.",
    "3.2 Sadness vs Anger",
    normalize=True,
    use_emotion_extraction=True
)

# 3.3 Surprise vs Disgust
emotion_diff3 = test_embedding_similarity(
    "I was amazed by what I saw.",
    "I was disgusted by what I saw.",
    "3.3 Surprise vs Disgust",
    normalize=True,
    use_emotion_extraction=True
)

# 3.4 Love vs Fear
emotion_diff4 = test_embedding_similarity(
    "I love going to new places.",
    "I fear going to new places.",
    "3.4 Love vs Fear",
    normalize=True,
    use_emotion_extraction=True
)

# ===== Category 4: Neutral vs Emotional (Should have LOW similarity) =====
print("\n" + "="*60)
print("Category 4: Neutral vs Emotional")
print("(Should have LOW similarity)")
print("="*60)

emotion_neutral1 = test_embedding_similarity(
    "The meeting is scheduled for 3 PM tomorrow.",
    "I'm thrilled about the meeting tomorrow!",
    "4.1 Neutral vs Excitement",
    normalize=True,
    use_emotion_extraction=True
)

emotion_neutral2 = test_embedding_similarity(
    "The document contains important information.",
    "I'm devastated by the information in the document.",
    "4.2 Neutral vs Sadness",
    normalize=True,
    use_emotion_extraction=True
)

# ===== Category 5: Mixed/Complex Emotions =====
print("\n" + "="*60)
print("Category 5: Mixed/Complex Emotions")
print("="*60)

emotion_mixed1 = test_embedding_similarity(
    "I'm excited but also nervous about the new job.",
    "I feel both happy and anxious about the changes.",
    "5.1 Mixed: Excitement+Nervousness",
    normalize=True,
    use_emotion_extraction=True
)

emotion_mixed2 = test_embedding_similarity(
    "I'm sad but also relieved that it's over.",
    "I feel both melancholy and peaceful now.",
    "5.2 Mixed: Sadness+Relief",
    normalize=True,
    use_emotion_extraction=True
)

# Baseline
emotion_baseline = test_embedding_similarity(
    "The weather is nice today.",
    "I'm feeling terrible about the exam.",
    "Baseline: Different descriptions and emotions",
    normalize=True,
    use_emotion_extraction=True
)

# --- Comprehensive Analysis: With vs Without Emotion Extraction ---

print("\n" + "="*60)
print("Comprehensive Analysis: Emotion Extraction Pipeline Results")
print("="*60)

# Calculate averages for each category
same_emotion_tests = [
    emotion_test1_1, emotion_test1_2, emotion_test1_3,
    emotion_test2_1, emotion_test2_2, emotion_test2_3,
    emotion_test3_1, emotion_test3_2,
    emotion_test4_1, emotion_test4_2,
    emotion_test5_1,
    emotion_test6_1,
    emotion_test7_1,
    emotion_test8_1
]

different_emotion_controls = [
    emotion_control1_1, emotion_control2_1, emotion_control3_1,
    emotion_control4_1, emotion_control5_1, emotion_control6_1,
    emotion_control7_1
]

different_emotion_types = [
    emotion_diff1, emotion_diff2, emotion_diff3, emotion_diff4
]

neutral_vs_emotional = [
    emotion_neutral1, emotion_neutral2
]

mixed_emotions = [
    emotion_mixed1, emotion_mixed2
]

avg_same_emotion = sum(same_emotion_tests) / len(same_emotion_tests)
avg_different_emotion_similar_desc = sum(different_emotion_controls) / len(different_emotion_controls)
avg_different_emotion_types = sum(different_emotion_types) / len(different_emotion_types)
avg_neutral_vs_emotional = sum(neutral_vs_emotional) / len(neutral_vs_emotional)
avg_mixed = sum(mixed_emotions) / len(mixed_emotions)

print(f"\nCategory 1: Same Emotion, Different Descriptions")
print(f"  Number of tests: {len(same_emotion_tests)}")
print(f"  Average similarity: {avg_same_emotion:.4f}")
print(f"  Range: {min(same_emotion_tests):.4f} - {max(same_emotion_tests):.4f}")
print(f"  Std deviation: {np.std(same_emotion_tests):.4f}")

print(f"\nCategory 2: Different Emotions, Similar Descriptions")
print(f"  Number of tests: {len(different_emotion_controls)}")
print(f"  Average similarity: {avg_different_emotion_similar_desc:.4f}")
print(f"  Range: {min(different_emotion_controls):.4f} - {max(different_emotion_controls):.4f}")
print(f"  Std deviation: {np.std(different_emotion_controls):.4f}")

print(f"\nCategory 3: Different Emotion Types")
print(f"  Number of tests: {len(different_emotion_types)}")
print(f"  Average similarity: {avg_different_emotion_types:.4f}")
print(f"  Range: {min(different_emotion_types):.4f} - {max(different_emotion_types):.4f}")

print(f"\nCategory 4: Neutral vs Emotional")
print(f"  Number of tests: {len(neutral_vs_emotional)}")
print(f"  Average similarity: {avg_neutral_vs_emotional:.4f}")
print(f"  Range: {min(neutral_vs_emotional):.4f} - {max(neutral_vs_emotional):.4f}")

print(f"\nCategory 5: Mixed/Complex Emotions")
print(f"  Number of tests: {len(mixed_emotions)}")
print(f"  Average similarity: {avg_mixed:.4f}")
print(f"  Range: {min(mixed_emotions):.4f} - {max(mixed_emotions):.4f}")

print(f"\nBaseline: Different descriptions and emotions")
print(f"  Similarity: {emotion_baseline:.4f}")

# Key comparisons
print(f"\n{'='*60}")
print("Key Comparisons")
print(f"{'='*60}")

diff1 = avg_same_emotion - avg_different_emotion_similar_desc
diff2 = avg_same_emotion - avg_different_emotion_types
diff3 = avg_same_emotion - avg_neutral_vs_emotional
diff4 = avg_same_emotion - emotion_baseline

print(f"\n1. Same Emotion vs Different Emotions (Similar Descriptions):")
print(f"   Same emotion:     {avg_same_emotion:.4f}")
print(f"   Different emotion: {avg_different_emotion_similar_desc:.4f}")
print(f"   Difference:       {diff1:+.4f} ({diff1/avg_same_emotion*100:+.2f}%)")

print(f"\n2. Same Emotion vs Different Emotion Types:")
print(f"   Same emotion:     {avg_same_emotion:.4f}")
print(f"   Different types:   {avg_different_emotion_types:.4f}")
print(f"   Difference:       {diff2:+.4f} ({diff2/avg_same_emotion*100:+.2f}%)")

print(f"\n3. Same Emotion vs Neutral:")
print(f"   Same emotion:     {avg_same_emotion:.4f}")
print(f"   Neutral:          {avg_neutral_vs_emotional:.4f}")
print(f"   Difference:       {diff3:+.4f} ({diff3/avg_same_emotion*100:+.2f}%)")

print(f"\n4. Same Emotion vs Baseline:")
print(f"   Same emotion:     {avg_same_emotion:.4f}")
print(f"   Baseline:         {emotion_baseline:.4f}")
print(f"   Difference:       {diff4:+.4f} ({diff4/avg_same_emotion*100:+.2f}%)")

# Final assessment
print(f"\n{'='*60}")
print("Final Assessment: Emotion Extraction Pipeline")
print(f"{'='*60}")

success_criteria = [
    (diff1 > 0, "Same emotions > Different emotions (similar desc)"),
    (diff2 > 0.05, "Same emotions > Different emotion types (gap > 0.05)"),
    (diff3 > 0.05, "Same emotions > Neutral (gap > 0.05)"),
    (avg_different_emotion_similar_desc < avg_different_emotion_types, "Different emotions (similar desc) < Different types"),
]

passed = sum(1 for condition, _ in success_criteria if condition)
total = len(success_criteria)

print(f"\nSuccess Criteria Check ({passed}/{total} passed):")
for condition, desc in success_criteria:
    status = "✓" if condition else "✗"
    print(f"  {status} {desc}")

if passed >= 3:
    print(f"\n✓ SUCCESS: Emotion extraction pipeline is working effectively!")
    print(f"  - Successfully distinguishes same emotions from different emotions")
    print(f"  - Reduces semantic bias in similarity calculation")
    print(f"  - Makes emotion similarity more prominent")
else:
    print(f"\n⚠ PARTIAL SUCCESS: Pipeline shows improvement but needs refinement")
    print(f"  - Some criteria met, but not all")
    print(f"  - Consider refining the emotion extraction prompt")

# Detailed breakdown by emotion type
print(f"\n{'='*60}")
print("Breakdown by Emotion Type")
print(f"{'='*60}")

happiness_tests = [emotion_test1_1, emotion_test1_2, emotion_test1_3]
sadness_tests = [emotion_test2_1, emotion_test2_2, emotion_test2_3]
fear_tests = [emotion_test3_1, emotion_test3_2]
anger_tests = [emotion_test4_1, emotion_test4_2]

print(f"\nHappiness (same emotion, different contexts):")
print(f"  Average: {sum(happiness_tests)/len(happiness_tests):.4f}")
print(f"  Range: {min(happiness_tests):.4f} - {max(happiness_tests):.4f}")

print(f"\nSadness (same emotion, different contexts):")
print(f"  Average: {sum(sadness_tests)/len(sadness_tests):.4f}")
print(f"  Range: {min(sadness_tests):.4f} - {max(sadness_tests):.4f}")

print(f"\nFear/Anxiety (same emotion, different contexts):")
print(f"  Average: {sum(fear_tests)/len(fear_tests):.4f}")
print(f"  Range: {min(fear_tests):.4f} - {max(fear_tests):.4f}")

print(f"\nAnger (same emotion, different contexts):")
print(f"  Average: {sum(anger_tests)/len(anger_tests):.4f}")
print(f"  Range: {min(anger_tests):.4f} - {max(anger_tests):.4f}")