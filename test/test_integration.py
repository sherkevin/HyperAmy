#!/usr/bin/env python
"""
测试整合后的 HippoRAG + 情感分析功能
"""
import sys
import os
import json

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sentiment'))

# 配置 API
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
    from api_config import PARALLEL_API_KEY, PARALLEL_BASE_URL, PARALLEL_MODEL_NAME, setup_api
    setup_api()
    llm_model_name = PARALLEL_MODEL_NAME
    llm_base_url = PARALLEL_BASE_URL
    embedding_model_name = 'GLM-Embedding-2'
    embedding_base_url = PARALLEL_BASE_URL
    print(f"✅ 使用并行智能云 API")
except ImportError:
    print("⚠️  使用默认配置")
    os.environ['OPENAI_API_KEY'] = 'sk-YaO0f0NsiW-drkCHbnDIHw'
    llm_model_name = 'GLM-4-Flash'
    llm_base_url = 'https://llmapi.paratera.com/v1'
    embedding_model_name = 'GLM-Embedding-2'
    embedding_base_url = 'https://llmapi.paratera.com/v1'

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
    "What makes people feel happy?",
    "What causes sadness?",
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

