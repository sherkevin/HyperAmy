# -*- coding: utf-8 -*-
"""
GoT 实验运行脚本 - 迷你版本（用于测试）
只处理前3个问题，快速验证流程
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING
from tqdm import tqdm
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None  # 避免类型检查错误
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

from sentence_transformers import SentenceTransformer
from llm import create_client
from llm.config import DEFAULT_MODEL, BETA_WARPING

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_qa_data(file_path: str) -> List[Dict]:
    """加载本能测试题数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_amygdala_chunks(file_path: str) -> List[Dict]:
    """加载带杏仁核特征的分块数据"""
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_faiss_index(chunks: List[Dict], embedding_model) -> Tuple[any, Dict[int, int]]:
    """构建 FAISS 索引"""
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available")
    
    sample_vec = np.array(chunks[0]['vector'])
    dim = len(sample_vec)
    
    index = faiss.IndexFlatIP(dim)
    
    vectors = []
    chunk_id_to_idx = {}
    
    for idx, chunk in enumerate(chunks):
        vec = np.array(chunk['vector']).astype('float32')
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        vectors.append(vec)
        chunk_id_to_idx[chunk['chunk_id']] = idx
    
    vectors_matrix = np.vstack(vectors).astype('float32')
    index.add(vectors_matrix)
    
    logger.info(f"Built FAISS index with {index.ntotal} vectors")
    return index, chunk_id_to_idx


def retrieve_baseline(query_vec: np.ndarray, 
                     faiss_index: any,
                     chunks: List[Dict],
                     chunk_id_to_idx: Dict[int, int],
                     num_to_retrieve: int = 3) -> Tuple[List[str], np.ndarray]:
    """标准 Cosine 检索（使用 FAISS）"""
    query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    query_vec_norm = query_vec_norm.astype('float32').reshape(1, -1)
    
    scores, indices = faiss_index.search(query_vec_norm, num_to_retrieve)
    
    retrieved_docs = []
    retrieved_scores = []
    
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            retrieved_docs.append(chunks[idx]['text'])
            retrieved_scores.append(float(score))
    
    return retrieved_docs, np.array(retrieved_scores)


def retrieve_hyperamy(query_vec: np.ndarray,
                     chunks: List[Dict],
                     num_to_retrieve: int = 3,
                     beta: float = BETA_WARPING) -> Tuple[List[str], np.ndarray]:
    """庞加莱畸变公式检索"""
    doc_vecs = np.array([chunk['vector'] for chunk in chunks])
    doc_masses = np.array([chunk.get('mass', 0.0) for chunk in chunks])
    doc_texts = [chunk['text'] for chunk in chunks]
    
    query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_vecs_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    
    cosine_similarities = np.dot(doc_vecs_norm, query_vec_norm).flatten()
    cosine_distances = 1.0 - cosine_similarities
    
    warped_distances = cosine_distances / (1.0 + beta * doc_masses)
    
    sorted_indices = np.argsort(warped_distances)
    top_indices = sorted_indices[:num_to_retrieve]
    
    retrieved_docs = [doc_texts[i] for i in top_indices]
    retrieved_scores = cosine_similarities[top_indices]
    
    return retrieved_docs, retrieved_scores


def generate_answer_with_llm(question: str, context: List[str], client, model_name: str = DEFAULT_MODEL) -> str:
    """使用 LLM 生成答案"""
    context_text = "\n\n".join([f"[文档 {i+1}]\n{doc}" for i, doc in enumerate(context)])
    
    prompt = f"""基于以下上下文文档回答问题。如果上下文中没有相关信息，请说明无法从提供的上下文中找到答案。

上下文：
{context_text}

问题：{question}

答案："""
    
    try:
        answer = client.get_answer(prompt, max_tokens=300, temperature=0.7, mode="normal")
        return answer.strip()
    except Exception as e:
        logger.error(f"Failed to generate answer: {e}")
        return f"[生成答案时出错: {e}]"


def run_mini_experiment(
    qa_file: str = "data/benchmarks/instinct_qa.json",
    chunks_file: str = "data/processed/got_amygdala.jsonl",
    output_file: str = "results/test_mini.json",
    num_questions: int = 3,  # 只处理前N个问题
    num_to_retrieve: int = 3,
    beta: float = BETA_WARPING,
    model_name: str = DEFAULT_MODEL
):
    """运行迷你实验（只处理少量问题用于测试）"""
    
    print("=" * 70)
    print("迷你实验 - 快速验证流程")
    print("=" * 70)
    
    # 1. 加载数据
    logger.info(f"Loading QA data from {qa_file}...")
    if not os.path.exists(qa_file):
        logger.error(f"QA文件不存在: {qa_file}")
        logger.info("提示: 需要先运行 src/gen_qa.py 生成测试题")
        return
    
    qa_pairs = load_qa_data(qa_file)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # 只取前N个
    qa_pairs = qa_pairs[:num_questions]
    logger.info(f"Processing first {len(qa_pairs)} questions for testing")
    
    logger.info(f"Loading chunks from {chunks_file}...")
    chunks = load_amygdala_chunks(chunks_file)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # 2. 构建 chunk_id 到 chunk 的映射
    chunk_dict = {chunk['chunk_id']: chunk for chunk in chunks}
    
    # 3. 初始化 embedding 模型
    logger.info("Loading embedding model...")
    print("Loading embedding model (this may take a moment)...", flush=True)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model loaded!", flush=True)
    
    # 4. 构建 FAISS 索引
    logger.info("Building FAISS index...")
    faiss_index, chunk_id_to_idx = build_faiss_index(chunks, embedding_model)
    print("✅ FAISS index built!", flush=True)
    
    # 5. 初始化 LLM 客户端
    logger.info(f"Initializing LLM client with model: {model_name}...")
    llm_client = create_client(model_name=model_name, mode="normal")
    print("✅ LLM client initialized!", flush=True)
    
    # 6. 运行实验
    logger.info("Running mini experiment...")
    print(f"\n处理 {len(qa_pairs)} 个问题...\n", flush=True)
    results = []
    
    for idx, qa_pair in enumerate(qa_pairs, 1):
        question = qa_pair['question']
        gold_answer = qa_pair.get('answer', qa_pair.get('ground_truth_answer', ''))
        gold_chunk_id = qa_pair.get('chunk_id', -1)
        
        print(f"\n[{idx}/{len(qa_pairs)}] 问题: {question[:60]}...", flush=True)
        
        # 获取查询向量
        query_vec = embedding_model.encode(question, convert_to_numpy=True)
        print("  ✓ 查询向量已编码", flush=True)
        
        # Group A: Oracle
        oracle_chunk = chunk_dict.get(gold_chunk_id)
        if oracle_chunk:
            oracle_context = [oracle_chunk['text']]
        else:
            oracle_context = ["[未找到对应的 chunk]"]
        
        print("  → 生成Oracle答案...", flush=True)
        oracle_answer = generate_answer_with_llm(question, oracle_context, llm_client, model_name)
        print(f"  ✓ Oracle答案生成完成 ({len(oracle_answer)} 字符)", flush=True)
        
        # Group B: Baseline
        print("  → Baseline检索...", flush=True)
        baseline_docs, baseline_scores = retrieve_baseline(
            query_vec, faiss_index, chunks, chunk_id_to_idx, num_to_retrieve
        )
        print("  → 生成Baseline答案...", flush=True)
        baseline_answer = generate_answer_with_llm(question, baseline_docs, llm_client, model_name)
        print(f"  ✓ Baseline答案生成完成 ({len(baseline_answer)} 字符)", flush=True)
        
        # Group C: HyperAmy
        print("  → HyperAmy检索...", flush=True)
        hyperamy_docs, hyperamy_scores = retrieve_hyperamy(
            query_vec, chunks, num_to_retrieve, beta
        )
        print("  → 生成HyperAmy答案...", flush=True)
        hyperamy_answer = generate_answer_with_llm(question, hyperamy_docs, llm_client, model_name)
        print(f"  ✓ HyperAmy答案生成完成 ({len(hyperamy_answer)} 字符)", flush=True)
        
        # 检查hit
        baseline_retrieved_chunk_ids = []
        for doc in baseline_docs:
            for chunk in chunks:
                if chunk['text'] == doc:
                    baseline_retrieved_chunk_ids.append(chunk['chunk_id'])
                    break
        
        hyperamy_retrieved_chunk_ids = []
        for doc in hyperamy_docs:
            for chunk in chunks:
                if chunk['text'] == doc:
                    hyperamy_retrieved_chunk_ids.append(chunk['chunk_id'])
                    break
        
        baseline_hit = gold_chunk_id in baseline_retrieved_chunk_ids
        hyperamy_hit = gold_chunk_id in hyperamy_retrieved_chunk_ids
        
        result = {
            'question': question,
            'gold_answer': gold_answer,
            'gold_chunk_id': gold_chunk_id,
            'oracle': {
                'context': oracle_context,
                'answer': oracle_answer
            },
            'baseline': {
                'context': baseline_docs,
                'scores': baseline_scores.tolist(),
                'answer': baseline_answer,
                'hit': baseline_hit,
                'retrieved_chunk_ids': baseline_retrieved_chunk_ids
            },
            'hyperamy': {
                'context': hyperamy_docs,
                'scores': hyperamy_scores.tolist(),
                'answer': hyperamy_answer,
                'hit': hyperamy_hit,
                'retrieved_chunk_ids': hyperamy_retrieved_chunk_ids
            }
        }
        
        results.append(result)
        print(f"  ✓ 问题 {idx} 处理完成 (Baseline hit: {baseline_hit}, HyperAmy hit: {hyperamy_hit})", flush=True)
    
    # 7. 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Mini experiment completed. Results saved to {output_file}")
    
    # 8. 打印统计信息
    baseline_hits = sum(1 for r in results if r['baseline']['hit'])
    hyperamy_hits = sum(1 for r in results if r['hyperamy']['hit'])
    
    print("\n" + "=" * 70)
    print("实验结果统计")
    print("=" * 70)
    print(f"总问题数: {len(results)}")
    print(f"Baseline hit rate: {baseline_hits}/{len(results)} ({100*baseline_hits/len(results) if results else 0:.1f}%)")
    print(f"HyperAmy hit rate: {hyperamy_hits}/{len(results)} ({100*hyperamy_hits/len(results) if results else 0:.1f}%)")
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GoT 迷你实验脚本（用于测试）")
    parser.add_argument("--qa", type=str, default="data/benchmarks/instinct_qa.json", help="测试题文件路径")
    parser.add_argument("--chunks", type=str, default="data/processed/got_amygdala.jsonl", help="分块数据文件路径")
    parser.add_argument("--output", type=str, default="results/test_mini.json", help="输出文件路径")
    parser.add_argument("--num", type=int, default=3, help="处理的问题数量（用于测试）")
    parser.add_argument("--k", type=int, default=3, help="检索的文档数量")
    parser.add_argument("--beta", type=float, default=BETA_WARPING, help="庞加莱畸变参数")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM 模型名称")
    
    args = parser.parse_args()
    
    run_mini_experiment(
        qa_file=args.qa,
        chunks_file=args.chunks,
        output_file=args.output,
        num_questions=args.num,
        num_to_retrieve=args.k,
        beta=args.beta,
        model_name=args.model
    )

