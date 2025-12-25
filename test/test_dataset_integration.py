#!/usr/bin/env python
"""
测试整合后的 HippoRAG + 情感分析功能，使用真实数据集
"""
import sys
import os
import json
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
print("HippoRAG + 情感分析 - 数据集测试")
print("=" * 70)

# 加载数据集（小样本测试）
dataset_name = 'hotpotqa'
dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'reproduce', 'dataset')
corpus_path = os.path.join(dataset_dir, f"{dataset_name}_corpus.json")
qa_path = os.path.join(dataset_dir, f"{dataset_name}.json")

print(f"\n【步骤 1】加载数据集...")
try:
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    print(f"   ✅ 语料库: {len(corpus)} 个文档")
    
    with open(qa_path, 'r') as f:
        samples = json.load(f)
    print(f"   ✅ QA 数据: {len(samples)} 个问题")
    
    # 只使用前 10 个文档和 3 个问题进行测试
    test_corpus = corpus[:10]
    test_samples = samples[:3]
    
    docs = [f"{doc['title']}\n{doc['text']}" for doc in test_corpus]
    queries = [s['question'] for s in test_samples]
    gold_answers = []
    for s in test_samples:
        if isinstance(s.get('answer'), list):
            gold_answers.append(s['answer'])
        else:
            gold_answers.append([s.get('answer', '')])
    
    print(f"   ✅ 测试规模: {len(docs)} 个文档, {len(queries)} 个问题")
    
except Exception as e:
    print(f"   ❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n【步骤 2】初始化增强版 HippoRAG...")
try:
    from sentiment.hipporag_enhanced import HippoRAGEnhanced
    from hipporag.utils.config_utils import BaseConfig
    
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', f'test_dataset_{dataset_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        embedding_base_url=embedding_base_url,
        force_index_from_scratch=False,  # 使用已有索引（如果有）
        retrieval_top_k=3,
    )
    
    hipporag = HippoRAGEnhanced(
        global_config=config,
        enable_sentiment=True,
        sentiment_weight=0.3,
        sentiment_model_name=llm_model_name
    )
    print(f"   ✅ HippoRAG 增强版初始化成功")
    
except Exception as e:
    print(f"   ❌ 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n【步骤 3】索引文档（包含情感分析）...")
try:
    print(f"   文档数量: {len(docs)}")
    print(f"   ⚠️  注意：情感向量提取可能需要较长时间...")
    hipporag.index(docs=docs)
    print(f"   ✅ 索引完成")
except Exception as e:
    print(f"   ❌ 索引失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n【步骤 4】测试检索和问答（情感增强）...")
try:
    print(f"   查询数量: {len(queries)}")
    qa_results, messages, metadata, retrieval_results, qa_eval_results = hipporag.rag_qa(
        queries=queries,
        gold_answers=gold_answers
    )
    
    print(f"   ✅ 完成！")
    
    # 显示评估结果
    if qa_eval_results:
        print(f"\n   评估结果:")
        print(f"     ExactMatch: {qa_eval_results.get('ExactMatch', 0):.4f}")
        print(f"     F1 Score: {qa_eval_results.get('F1', 0):.4f}")
    
    # 显示答案
    print(f"\n   问答结果:")
    for i, (query, result) in enumerate(zip(queries, qa_results), 1):
        print(f"\n   问题 {i}: {query}")
        print(f"   答案: {result.answer}")
        if result.gold_answers:
            print(f"   标准答案: {result.gold_answers}")
    
except Exception as e:
    print(f"   ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ 数据集测试完成！")
print("=" * 70)

