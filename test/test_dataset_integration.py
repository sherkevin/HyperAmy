#!/usr/bin/env python
"""
测试整合后的 HippoRAG + 情感分析功能，使用真实数据集
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
        enable_emotion=True,
        emotion_weight=0.3,
        emotion_model_name=llm_model_name
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

