#!/usr/bin/env python
"""
测试整合后的 HippoRAG + 情感分析功能
"""
import sys
import os

# 配置 API - 使用 dotenv 和 config 模块
from llm.config import API_KEY, BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS

# 设置环境变量（HippoRAG 需要 OPENAI_API_KEY）
os.environ['OPENAI_API_KEY'] = API_KEY

# 模型配置（可以在这里自定义，或使用默认值）
llm_model_name = DEFAULT_MODEL  # 可以修改为其他模型名称
llm_base_url = BASE_URL
# HippoRAG 支持多种嵌入模型格式：
# - "text-embedding-xxx" 使用 OpenAIEmbeddingModel
# - "VLLM/xxx" 使用 VLLMEmbeddingModel（直接传递模型名称给 API）
# 由于 API 期望的模型名称是 "GLM-Embedding-2"，使用 VLLM 前缀
embedding_model_name = f"VLLM/{DEFAULT_EMBEDDING_MODEL}"  # 可以修改为其他嵌入模型名称
embedding_base_url = API_URL_EMBEDDINGS  # 使用完整的 embeddings API URL

print(f"✅ 使用配置:")
print(f"   API_KEY: {API_KEY[:10]}...")
print(f"   BASE_URL: {BASE_URL}")
print(f"   LLM Model: {llm_model_name}")
print(f"   Embedding Model: {embedding_model_name}")

print("=" * 70)
print("HippoRAG + 情感分析整合测试")
print("=" * 70)

# 测试数据
test_docs = [
    "I'm thrilled about winning the competition! This is the happiest day of my life.",
    "I'm devastated by the loss of my pet. Everything feels hopeless now.",
    "The weather is nice today. It's a beautiful sunny day.",
]

test_queries = [
    "What makes you feel happy?",
    "What makes you feel sadness?",
]

print(f"\n【步骤 1】初始化增强版 HippoRAG...")
try:
    from sentiment.hipporag_enhanced import HippoRAGEnhanced
    from hipporag.utils.config_utils import BaseConfig
    
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'test_integration')
    os.makedirs(save_dir, exist_ok=True)
    
    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        embedding_base_url=embedding_base_url,
        force_index_from_scratch=True,
        retrieval_top_k=2,
    )
    
    hipporag = HippoRAGEnhanced(
        global_config=config,
        enable_emotion=True,
        emotion_weight=0.3,
        emotion_model_name=llm_model_name
    )
    print(f"   ✅ HippoRAG 增强版初始化成功")
    print(f"   情感分析: {'启用' if hipporag.enable_emotion else '禁用'}")
    
except Exception as e:
    print(f"   ❌ 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n【步骤 2】索引文档（包含情感分析）...")
try:
    print(f"   文档数量: {len(test_docs)}")
    hipporag.index(docs=test_docs)
    print(f"   ✅ 索引完成")
except Exception as e:
    print(f"   ❌ 索引失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n【步骤 3】测试检索（情感增强）...")
try:
    print(f"   查询数量: {len(test_queries)}")
    results = hipporag.retrieve(queries=test_queries, num_to_retrieve=2)
    
    print(f"   ✅ 检索完成")
    for i, (query, result) in enumerate(zip(test_queries, results), 1):
        print(f"\n   问题 {i}: {query}")
        print(f"   检索到的文档数: {len(result.docs)}")
        for j, (doc, score) in enumerate(zip(result.docs, result.doc_scores), 1):
            print(f"     文档 {j} (分数: {score:.4f}): {doc[:60]}...")
    
except Exception as e:
    print(f"   ❌ 检索失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n【步骤 4】测试问答...")
try:
    qa_results, messages, metadata = hipporag.rag_qa(queries=test_queries)
    print(f"   ✅ 问答完成")
    for i, (query, result) in enumerate(zip(test_queries, qa_results), 1):
        print(f"\n   问题 {i}: {query}")
        print(f"   答案: {result.answer}")
except Exception as e:
    print(f"   ❌ 问答失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ 整合测试完成！")
print("=" * 70)

