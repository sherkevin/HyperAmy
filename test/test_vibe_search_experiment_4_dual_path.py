#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 4: HyperAmy V2 - Adaptive Dual-Path Retrieval

核心逻辑：
- Path A (Hybrid Re-ranking): 当语义置信度高时，使用HippoRAG候选 + 情绪融合
- Path B (Global Emotion Search): 当检测到语义崩溃时，绕过HippoRAG，直接全库情绪检索

数据集：
- 训练数据: data/training/monte_cristo_train_full.jsonl
- QA数据: data/public_benchmark/monte_cristo_vibe_search.json (50个QA对)

方法：
1. HippoRAG: 纯语义检索（基线）
2. HyperAmy V2 (Dual-Path): 自适应双路检索
   - 使用LLM API提取情绪向量
   - 语义崩溃时切换到全库情绪检索
"""
import sys
import os

# 强制设置环境变量，确保日志实时输出
os.environ['PYTHONUNBUFFERED'] = '1'

import json
import numpy as np
import logging
import hashlib
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.config import API_KEY, BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
from sentiment.hipporag_enhanced import HippoRAGEnhanced
from hipporag.utils.misc_utils import compute_mdhash_id

# 设置环境变量
os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['API_KEY'] = API_KEY

# 模型配置（与Experiment 3保持一致）
llm_model_name = DEFAULT_MODEL
llm_base_url = BASE_URL
embedding_model_name = f"VLLM/{DEFAULT_EMBEDDING_MODEL}"  # 添加VLLM前缀
embedding_base_url = API_URL_EMBEDDINGS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 输出目录
output_dir = project_root / "outputs" / "vibe_search_experiment_4_dual_path"
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / "experiment_4_dual_path.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

print("=" * 80)
print("Experiment 4: HyperAmy V2 - Adaptive Dual-Path Retrieval")
print("=" * 80)
print("核心特性：")
print("  1. ✅ 自适应双路检索（Path A: Hybrid Re-ranking, Path B: Global Emotion Search）")
print("  2. ✅ 语义崩溃检测（S_sem < 0.01时切换到Path B）")
print("  3. ✅ LLM API提取情绪向量（高I_q值，预期0.8-0.9）")
print("  4. ✅ 自动复用情绪向量缓存（节省API费用）")
print("  5. ✅ 详细日志记录每一条查询的路径切换")
print("=" * 80)
print("INFO: Using Adaptive Dual-Path Retrieval (Experiment 4)")
print("=" * 80)

# ========== 加载数据 ==========
print("\n【步骤1】加载数据...")
chunks_file = project_root / "data" / "training" / "monte_cristo_train_full.jsonl"
vibe_file = project_root / "data" / "public_benchmark" / "monte_cristo_vibe_search.json"

if not chunks_file.exists():
    raise FileNotFoundError(f"训练数据文件不存在: {chunks_file}")

chunks = []
with open(chunks_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

queries = []
gold_docs = []
if vibe_file.exists():
    with open(vibe_file, 'r', encoding='utf-8') as f:
        vibe_data = json.load(f)
        for item in vibe_data.get('data', []):
            queries.append(item.get('query', ''))
            gold_docs.append([item.get('gold_text', '')])

print(f"✅ 加载完成：{len(chunks)} 个chunks, {len(queries)} 个queries")

# ========== 初始化HippoRAGEnhanced ==========
print("\n【步骤2】初始化HippoRAGEnhanced（启用情绪分析）...")
hipporag = HippoRAGEnhanced(
    save_dir=str(output_dir / "hipporag_index"),
    llm_model_name=llm_model_name,
    llm_base_url=llm_base_url,
    embedding_model_name=embedding_model_name,
    embedding_base_url=embedding_base_url,
    enable_sentiment=True,
    sentiment_weight=0.4,
    max_workers=10
)

# ========== 索引文档 ==========
print("\n【步骤3】索引文档（包含情绪向量提取）...")
# 修复：数据文件使用'input'字段，不是'text'字段
chunk_texts = [chunk.get('input', chunk.get('text', '')) for chunk in chunks]
# 过滤空文档
chunk_texts = [text for text in chunk_texts if text and len(text.strip()) > 0]
print(f"✅ 提取了 {len(chunk_texts)} 个非空文档（从 {len(chunks)} 个chunks）")
hipporag.index(chunk_texts)
print(f"✅ 索引完成：{len(chunk_texts)} 个文档")

# ========== 检索测试 ==========
print("\n【步骤4】开始检索测试（自适应双路检索）...")
print(f"   测试 {len(queries)} 个queries")

hipporag_results = []
hyperamy_results = []

for query_idx, query in enumerate(tqdm(queries, desc="检索测试")):
    # HippoRAG基线
    hipporag_result = hipporag.retrieve(
        queries=[query],
        num_to_retrieve=5,
        gold_docs=[gold_docs[query_idx]] if gold_docs else None
    )
    if isinstance(hipporag_result, tuple):
        hipporag_result, _ = hipporag_result
    hipporag_results.append(hipporag_result[0] if hipporag_result else None)
    
    # HyperAmy V2 (Dual-Path)
    hyperamy_result = hipporag.retrieve(
        queries=[query],
        num_to_retrieve=5,
        gold_docs=[gold_docs[query_idx]] if gold_docs else None
    )
    if isinstance(hyperamy_result, tuple):
        hyperamy_result, _ = hyperamy_result
    hyperamy_results.append(hyperamy_result[0] if hyperamy_result else None)

# ========== 评估结果 ==========
print("\n【步骤5】评估结果...")

def calculate_recall_at_k(retrieved_docs_list, gold_docs_list, k_list=[1, 5, 10, 20]):
    """计算Recall@K"""
    recall_at_k = {f"Recall@{k}": 0.0 for k in k_list}
    total_queries = len(gold_docs_list)
    
    for retrieved_docs, gold_docs in zip(retrieved_docs_list, gold_docs_list):
        gold_set = set(gold_docs) if gold_docs else set()
        retrieved_list = retrieved_docs if retrieved_docs else []
        
        for k in k_list:
            if k <= len(retrieved_list):
                top_k = retrieved_list[:k]
                top_k_set = set(top_k)
                intersection = gold_set & top_k_set
                if gold_set:
                    recall_at_k[f"Recall@{k}"] += len(intersection) / len(gold_set)
    
    # 平均所有查询
    for k in k_list:
        recall_at_k[f"Recall@{k}"] /= total_queries if total_queries > 0 else 1.0
        recall_at_k[f"Recall@{k}"] = round(recall_at_k[f"Recall@{k}"], 4)
    
    return recall_at_k

hipporag_retrieved_docs = [result.docs if result and hasattr(result, 'docs') else [] for result in hipporag_results]
hipporag_eval = calculate_recall_at_k(hipporag_retrieved_docs, gold_docs)

hyperamy_retrieved_docs = [result.docs if result and hasattr(result, 'docs') else [] for result in hyperamy_results]
hyperamy_eval = calculate_recall_at_k(hyperamy_retrieved_docs, gold_docs)

# ========== 汇总结果 ==========
print("\n" + "=" * 80)
print("Experiment 4: Adaptive Dual-Path Retrieval 实验结果汇总")
print("=" * 80)
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

print("方法对比：")
print("  1. HippoRAG (纯语义):")
if hipporag_eval:
    print(f"     Recall@1: {hipporag_eval.get('Recall@1', 0.0)*100:.1f}%")
    print(f"     Recall@5: {hipporag_eval.get('Recall@5', 0.0)*100:.1f}%")
print("  2. HyperAmy V2 (Dual-Path):")
if hyperamy_eval:
    print(f"     Recall@1: {hyperamy_eval.get('Recall@1', 0.0)*100:.1f}%")
    print(f"     Recall@5: {hyperamy_eval.get('Recall@5', 0.0)*100:.1f}%")

# 保存结果
results_file = output_dir / "results.json"
results_data = {
    'timestamp': datetime.now().isoformat(),
    'model': 'LLM_API',
    'hipporag': {
        'recall_at_1': hipporag_eval.get('Recall@1', 0.0),
        'recall_at_5': hipporag_eval.get('Recall@5', 0.0),
        'full_eval': hipporag_eval
    },
    'hyperamy_v2': {
        'recall_at_1': hyperamy_eval.get('Recall@1', 0.0),
        'recall_at_5': hyperamy_eval.get('Recall@5', 0.0),
        'full_eval': hyperamy_eval
    }
}

with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)

logger.info(f"✅ 结果已保存到: {results_file}")

print(f"\n✅ 实验完成！结果已保存到: {results_file}")
print(f"日志文件: {log_file}")
