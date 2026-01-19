#!/usr/bin/env python
"""验证数据集"""
import json
import os

dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'reproduce', 'dataset')
qa_path = os.path.join(dataset_dir, "hotpotqa.json")
corpus_path = os.path.join(dataset_dir, "hotpotqa_corpus.json")

print("=" * 70)
print("数据集验证")
print("=" * 70)

# 验证 QA 数据
with open(qa_path, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)
print(f"\n✅ QA 数据:")
print(f"   问题数量: {len(qa_data)}")
print(f"   第一个问题: {qa_data[0]['question'][:60]}...")
print(f"   第一个答案: {qa_data[0]['answer']}")
print(f"   相关文档数: {len(qa_data[0].get('relevant_docs', []))}")

# 验证语料库
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = json.load(f)
print(f"\n✅ 语料库:")
print(f"   文档数量: {len(corpus)}")
print(f"   第一个文档标题: {corpus[0]['title'][:60]}...")
print(f"   第一个文档文本长度: {len(corpus[0]['text'])} 字符")

# 文件大小
qa_size = os.path.getsize(qa_path) / (1024 * 1024)
corpus_size = os.path.getsize(corpus_path) / (1024 * 1024)
print(f"\n✅ 文件大小:")
print(f"   hotpotqa.json: {qa_size:.2f} MB")
print(f"   hotpotqa_corpus.json: {corpus_size:.2f} MB")

print(f"\n✅ 数据集验证完成！")

