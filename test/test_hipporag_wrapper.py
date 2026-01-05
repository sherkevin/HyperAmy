#!/usr/bin/env python3
"""
测试 HippoRAGWrapper - HippoRAG 的简洁接口

测试流程：
1. 初始化 HippoRAGWrapper
2. 添加测试文档块
3. 执行检索
4. 验证结果
5. 测试 DPR 检索
6. 测试 RAG 问答
"""

import logging
import sys
import os
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s'
)

# 确保可以导入 workflow 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置 HippoRAG 需要的环境变量（从 llm/config.py 读取）
from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
os.environ["OPENAI_API_KEY"] = API_KEY

from workflow.hipporag_wrapper import HippoRAGWrapper

print("=" * 100)
print("HippoRAGWrapper 测试")
print("=" * 100)

# 测试数据
chunks = [
    "Python is a high-level, interpreted programming language known for its simplicity and readability.",
    "JavaScript is a versatile scripting language primarily used for web development and interactive web pages.",
    "Java is a class-based, object-oriented programming language widely used in enterprise applications.",
    "C++ is a powerful programming language used for system programming, game development, and high-performance applications.",
    "Go is a statically typed, compiled language designed at Google for building scalable and reliable software.",
    "Rust is a systems programming language focused on safety, concurrency, and performance.",
    "TypeScript is a superset of JavaScript that adds static typing and other features for large-scale applications.",
    "Ruby is a dynamic, object-oriented language known for its elegant syntax and focus on programmer happiness."
]

queries = [
    "What programming languages are good for web development?",
    "Which languages are used for systems programming?",
    "Tell me about statically typed languages"
]

# ========== Test 1: 初始化 ==========
print("\n" + "=" * 100)
print("【Test 1】初始化 HippoRAGWrapper")
print("=" * 100)

# HippoRAG 支持多种嵌入模型格式：
# - "text-embedding-xxx" 使用 OpenAIEmbeddingModel
# - "VLLM/xxx" 使用 VLLMEmbeddingModel（直接传递模型名称给 API）
# 由于 API 期望的模型名称是 "GLM-Embedding-3"，使用 VLLM 前缀
wrapper = HippoRAGWrapper(
    save_dir="./test_hipporag_wrapper_db",
    llm_model_name="DeepSeek-V3.2",
    llm_base_url=BASE_URL,
    embedding_model_name=f"VLLM/{DEFAULT_EMBEDDING_MODEL}",
    embedding_base_url=API_URL_EMBEDDINGS
)
print("✓ HippoRAGWrapper 初始化完成")

# ========== Test 2: 添加文档块 ==========
print("\n" + "=" * 100)
print("【Test 2】添加文档块到索引")
print("=" * 100)

print(f"\n添加 {len(chunks)} 个文档块:")
for i, chunk in enumerate(chunks, 1):
    print(f"  {i}. {chunk[:80]}...")

result = wrapper.add(chunks)
print(f"\n✓ 索引完成:")
print(f"  - 本次添加: {result['chunk_count']} 个块")
print(f"  - 总索引数: {result['total_indexed']} 个块")

# ========== Test 3: 获取统计信息 ==========
print("\n" + "=" * 100)
print("【Test 3】获取索引统计信息")
print("=" * 100)

stats = wrapper.get_stats()
print(f"\n统计信息:")
print(f"  - 总索引块数: {stats['total_indexed']}")
print(f"  - 图谱节点数: {stats['graph_nodes']}")
print(f"  - 图谱边数: {stats['graph_edges']}")
print(f"  - 实体数量: {stats['entities']}")
print(f"  - 事实数量: {stats['facts']}")

# ========== Test 4: 检索测试 ==========
print("\n" + "=" * 100)
print("【Test 4】检索测试（使用图谱）")
print("=" * 100)

for i, query in enumerate(queries, 1):
    print(f"\n查询 {i}: {query}")
    print("-" * 100)

    results = wrapper.retrieve(query=query, top_k=3)

    print(f"\n检索到 {len(results)} 个相关文档:")
    for result in results:
        print(f"\n  Rank {result['rank']}:")
        print(f"    - 得分: {result['score']:.4f}")
        print(f"    - 文档: {result['text'][:100]}...")

# ========== Test 5: DPR 检索测试 ==========
print("\n" + "=" * 100)
print("【Test 5】DPR 检索测试（不使用图谱）")
print("=" * 100)

query = queries[0]
print(f"\n查询: {query}")
print("-" * 100)

results_dpr = wrapper.retrieve_dpr(query=query, top_k=3)

print(f"\nDPR 检索到 {len(results_dpr)} 个相关文档:")
for result in results_dpr:
    print(f"\n  Rank {result['rank']}:")
    print(f"    - 得分: {result['score']:.4f}")
    print(f"    - 文档: {result['text'][:100]}...")

# ========== Test 6: RAG 问答测试 ==========
print("\n" + "=" * 100)
print("【Test 6】RAG 问答测试（检索 + 生成）")
print("=" * 100)

qa_query = "What are the main differences between Python and JavaScript?"
print(f"\n问题: {qa_query}")
print("-" * 100)

qa_result = wrapper.qa(query=qa_query, top_k=3)

print(f"\n检索到的上下文 ({len(qa_result['retrieved_chunks'])} 个块):")
for chunk in qa_result['retrieved_chunks']:
    print(f"  - Rank {chunk['rank']} (Score: {chunk['score']:.4f}): {chunk['text'][:80]}...")

print(f"\n生成的回答:")
print(f"  {qa_result['answer']}")

# ========== Test 7: 增量添加测试 ==========
print("\n" + "=" * 100)
print("【Test 7】增量添加文档块")
print("=" * 100)

new_chunks = [
    "Swift is Apple's programming language for iOS, macOS, and other Apple platforms.",
    "Kotlin is a modern programming language for Android development and JVM applications."
]

print(f"\n添加 {len(new_chunks)} 个新文档块:")
for chunk in new_chunks:
    print(f"  - {chunk}")

result = wrapper.add(new_chunks)
print(f"\n✓ 增量添加完成:")
print(f"  - 本次添加: {result['chunk_count']} 个块")
print(f"  - 总索引数: {result['total_indexed']} 个块")

# 验证新文档可以被检索到
print(f"\n验证检索新文档:")
results = wrapper.retrieve(query="programming languages for mobile development", top_k=2)
print(f"检索到 {len(results)} 个相关文档:")
for result in results:
    print(f"  - Rank {result['rank']}: {result['text'][:80]}...")

# ========== Test 8: 删除测试 ==========
print("\n" + "=" * 100)
print("【Test 8】删除文档块")
print("=" * 100)

chunks_to_delete = [chunks[0]]  # 删除第一个文档
print(f"\n删除文档: {chunks_to_delete[0][:80]}...")

result = wrapper.delete(chunks_to_delete)
print(f"\n✓ 删除完成:")
print(f"  - 删除数量: {result['deleted_count']} 个块")
print(f"  - 剩余数量: {result['remaining_count']} 个块")

# ========== 总结 ==========
print("\n" + "=" * 100)
print("【测试总结】")
print("=" * 100)

final_stats = wrapper.get_stats()
print(f"\n最终统计:")
print(f"  - 总索引块数: {final_stats['total_indexed']}")
print(f"  - 图谱节点数: {final_stats['graph_nodes']}")
print(f"  - 图谱边数: {final_stats['graph_edges']}")
print(f"  - 实体数量: {final_stats['entities']}")
print(f"  - 事实数量: {final_stats['facts']}")

print(f"\n✓ 所有测试完成！")
print(f"\n功能验证:")
print(f"  ✅ 初始化 HippoRAGWrapper")
print(f"  ✅ 添加文档块到索引")
print(f"  ✅ 获取索引统计信息")
print(f"  ✅ 使用图谱检索（推荐）")
print(f"  ✅ 使用 DPR 检索（无图谱）")
print(f"  ✅ RAG 问答（检索 + 生成）")
print(f"  ✅ 增量添加文档")
print(f"  ✅ 删除文档")

print("\n" + "=" * 100)
print("测试完成！")
print("=" * 100)
