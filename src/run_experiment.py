# -*- coding: utf-8 -*-
"""
GoT 实验运行脚本

实现三组对比实验：
- Group A (Oracle): 直接使用正确答案的 Chunk
- Group B (Baseline): 标准 Cosine 检索（使用 FAISS）
- Group C (HyperAmy): 庞加莱畸变公式检索
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

from sentence_transformers import SentenceTransformer
from llm import create_client
from llm.config import DEFAULT_MODEL, BETA_WARPING
from sentiment.hipporag_enhanced import HippoRAGEnhanced

logging.basicConfig(level=logging.INFO)
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


def build_faiss_index(chunks: List[Dict], embedding_model) -> Tuple[faiss.Index, Dict[int, int]]:
    """
    构建 FAISS 索引
    
    Args:
        chunks: 分块列表
        embedding_model: SentenceTransformer 模型
        
    Returns:
        Tuple[faiss.Index, Dict]: (FAISS 索引, chunk_id 到索引位置的映射)
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
    
    # 获取向量维度
    sample_vec = np.array(chunks[0]['vector'])
    dim = len(sample_vec)
    
    # 创建 FAISS 索引（使用内积，需要归一化向量）
    index = faiss.IndexFlatIP(dim)
    
    # 构建向量矩阵和映射
    vectors = []
    chunk_id_to_idx = {}
    
    for idx, chunk in enumerate(chunks):
        vec = np.array(chunk['vector']).astype('float32')
        # 归一化向量（FAISS 内积需要归一化）
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        vectors.append(vec)
        chunk_id_to_idx[chunk['chunk_id']] = idx
    
    # 添加到索引
    vectors_matrix = np.vstack(vectors).astype('float32')
    index.add(vectors_matrix)
    
    logger.info(f"Built FAISS index with {index.ntotal} vectors")
    
    return index, chunk_id_to_idx


def retrieve_baseline(query_vec: np.ndarray, 
                     faiss_index: faiss.Index,
                     chunks: List[Dict],
                     chunk_id_to_idx: Dict[int, int],
                     num_to_retrieve: int = 3) -> Tuple[List[str], np.ndarray]:
    """
    标准 Cosine 检索（使用 FAISS）
    
    Args:
        query_vec: 查询向量
        faiss_index: FAISS 索引
        chunks: 分块列表
        chunk_id_to_idx: chunk_id 到索引位置的映射
        num_to_retrieve: 要检索的数量
        
    Returns:
        Tuple[List[str], np.ndarray]: (检索到的文档文本列表, 对应的分数数组)
    """
    # 归一化查询向量
    query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    query_vec_norm = query_vec_norm.astype('float32').reshape(1, -1)
    
    # 搜索
    scores, indices = faiss_index.search(query_vec_norm, num_to_retrieve)
    
    # 获取文档
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
    """
    庞加莱畸变公式检索
    
    Args:
        query_vec: 查询向量
        chunks: 分块列表（包含 vector 和 mass）
        num_to_retrieve: 要检索的数量
        beta: 畸变参数
        
    Returns:
        Tuple[List[str], np.ndarray]: (检索到的文档文本列表, 对应的分数数组)
    """
    # 准备数据
    doc_vecs = np.array([chunk['vector'] for chunk in chunks])
    doc_masses = np.array([chunk.get('mass', 0.0) for chunk in chunks])
    doc_texts = [chunk['text'] for chunk in chunks]
    
    # 归一化向量
    query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_vecs_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    
    # 计算余弦相似度
    cosine_similarities = np.dot(doc_vecs_norm, query_vec_norm)
    
    # 转换为距离
    cosine_distances = 1.0 - cosine_similarities
    
    # 应用庞加莱畸变公式: warped_dist = dist / (1 + beta * mass)
    warped_distances = cosine_distances / (1.0 + beta * doc_masses)
    
    # 排序（距离越小越好）
    sorted_indices = np.argsort(warped_distances)
    
    # 获取 top-k
    top_indices = sorted_indices[:num_to_retrieve]
    
    # 返回文档和分数
    retrieved_docs = [doc_texts[i] for i in top_indices]
    retrieved_scores = cosine_similarities[top_indices]
    
    return retrieved_docs, retrieved_scores


def generate_answer_with_llm(question: str, context: List[str], client, model_name: str = DEFAULT_MODEL, max_retries: int = 3) -> str:
    """
    使用 LLM 生成答案（带重试机制）
    
    Args:
        question: 问题
        context: 上下文文档列表
        client: LLM 客户端
        model_name: 模型名称
        max_retries: 最大重试次数
        
    Returns:
        str: 生成的答案
    """
    import time
    
    context_text = "\n\n".join([f"[文档 {i+1}]\n{doc}" for i, doc in enumerate(context)])
    
    prompt = f"""基于以下上下文文档回答问题。如果上下文中没有相关信息，请说明无法从提供的上下文中找到答案。

上下文：
{context_text}

问题：{question}

答案："""
    
    # 重试机制：指数退避
    for attempt in range(max_retries):
        try:
            answer = client.get_answer(prompt, max_tokens=300, temperature=0.7, mode="normal")
            return answer.strip()
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            
            # 如果是最后一次尝试，返回错误
            if attempt == max_retries - 1:
                logger.error(f"Failed to generate answer after {max_retries} attempts: {e}")
                return f"[生成答案时出错: {e}]"
            
            # 指数退避：1s, 2s, 4s
            wait_time = 2 ** attempt
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # 理论上不会到达这里
    return f"[生成答案时出错: 未知错误]"


def run_experiment(
    qa_file: str = "data/benchmarks/instinct_qa.json",
    chunks_file: str = "data/processed/got_amygdala.jsonl",
    output_file: str = "results/experiment_run_v1.json",
    num_to_retrieve: int = 3,
    beta: float = BETA_WARPING,
    model_name: str = DEFAULT_MODEL
):
    """
    运行完整实验
    
    Args:
        qa_file: 测试题文件路径
        chunks_file: 分块数据文件路径
        output_file: 输出文件路径
        num_to_retrieve: 每个问题检索的文档数量
        beta: 庞加莱畸变参数
        model_name: LLM 模型名称
    """
    # 1. 加载数据
    logger.info(f"Loading QA data from {qa_file}...")
    qa_pairs = load_qa_data(qa_file)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    logger.info(f"Loading chunks from {chunks_file}...")
    chunks = load_amygdala_chunks(chunks_file)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # 2. 构建 chunk_id 到 chunk 的映射
    chunk_dict = {chunk['chunk_id']: chunk for chunk in chunks}
    
    # 3. 初始化 embedding 模型
    logger.info("Loading embedding model...")
    print("Loading embedding model...", flush=True)  # 强制输出
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded!", flush=True)
    
    # 4. 构建 FAISS 索引（用于 Baseline）
    logger.info("Building FAISS index for baseline retrieval...")
    faiss_index, chunk_id_to_idx = build_faiss_index(chunks, embedding_model)
    
    # 5. 初始化 LLM 客户端
    logger.info(f"Initializing LLM client with model: {model_name}...")
    llm_client = create_client(model_name=model_name, mode="normal")
    
    # 6. 运行实验
    logger.info("Running experiment...")
    print("Starting experiment...", flush=True)
    results = []
    
    # 每处理5个问题保存一次中间结果
    save_interval = 5
    
    for idx, qa_pair in enumerate(tqdm(qa_pairs, desc="Processing questions")):
        question = qa_pair['question']
        gold_answer = qa_pair['answer']
        gold_chunk_id = qa_pair['chunk_id']
        
        logger.info(f"Processing question {idx+1}/{len(qa_pairs)}: {question[:50]}...")
        print(f"Processing question {idx+1}/{len(qa_pairs)}", flush=True)
        
        # 获取查询向量
        query_vec = embedding_model.encode(question, convert_to_numpy=True)
        print(f"Query vector encoded", flush=True)
        
        # Group A: Oracle（直接使用正确答案的 Chunk）
        oracle_chunk = chunk_dict.get(gold_chunk_id)
        if oracle_chunk:
            oracle_context = [oracle_chunk['text']]
        else:
            oracle_context = ["[未找到对应的 chunk]"]
        
        oracle_answer = generate_answer_with_llm(question, oracle_context, llm_client, model_name)
        
        # Group B: Baseline（标准 Cosine 检索）
        baseline_docs, baseline_scores = retrieve_baseline(
            query_vec, faiss_index, chunks, chunk_id_to_idx, num_to_retrieve
        )
        baseline_answer = generate_answer_with_llm(question, baseline_docs, llm_client, model_name)
        
        # Group C: HyperAmy（庞加莱畸变公式检索）
        hyperamy_docs, hyperamy_scores = retrieve_hyperamy(
            query_vec, chunks, num_to_retrieve, beta
        )
        hyperamy_answer = generate_answer_with_llm(question, hyperamy_docs, llm_client, model_name)
        
        # 检查是否检索到正确答案的 chunk
        baseline_hit = gold_chunk_id in [chunks[chunk_id_to_idx.get(cid, -1)]['chunk_id'] 
                                         for cid in range(len(chunks)) 
                                         if chunk_id_to_idx.get(cid, -1) >= 0][:num_to_retrieve]
        hyperamy_hit = gold_chunk_id in [chunk['chunk_id'] for chunk in chunks 
                                        if chunk['chunk_id'] in [c['chunk_id'] for c in chunks]][:num_to_retrieve]
        
        # 简化：检查 gold_chunk_id 是否在检索到的文档对应的 chunk_id 中
        # 需要从检索到的文档反推 chunk_id
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
        
        # 每处理save_interval个问题保存一次中间结果
        if (idx + 1) % save_interval == 0:
            logger.info(f"Saving intermediate results after {idx + 1} questions...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Intermediate results saved: {len(results)} questions processed")
    
    # 7. 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Experiment completed. Results saved to {output_file}")
    
    # 8. 打印统计信息
    baseline_hits = sum(1 for r in results if r['baseline']['hit'])
    hyperamy_hits = sum(1 for r in results if r['hyperamy']['hit'])
    
    logger.info(f"Statistics:")
    logger.info(f"  Total questions: {len(results)}")
    logger.info(f"  Baseline hit rate: {baseline_hits}/{len(results)} ({100*baseline_hits/len(results):.1f}%)")
    logger.info(f"  HyperAmy hit rate: {hyperamy_hits}/{len(results)} ({100*hyperamy_hits/len(results):.1f}%)")
    
    # 9. 发送邮件通知（如果配置了邮箱）
    try:
        from email_notifier import notify_experiment_completion
        recipient_email = os.getenv("EMAIL_RECIPIENT")
        if recipient_email:
            is_test = "test" in output_file.lower()
            notify_experiment_completion(output_file, recipient_email, is_test=is_test)
        else:
            logger.info("EMAIL_RECIPIENT not set, skipping email notification")
    except Exception as e:
        logger.warning(f"Failed to send email notification: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GoT 实验运行脚本")
    parser.add_argument("--qa", type=str, default="data/benchmarks/instinct_qa.json", help="测试题文件路径")
    parser.add_argument("--chunks", type=str, default="data/processed/got_amygdala.jsonl", help="分块数据文件路径")
    parser.add_argument("--output", type=str, default="results/experiment_run_v1.json", help="输出文件路径")
    parser.add_argument("--k", type=int, default=3, help="检索的文档数量")
    parser.add_argument("--beta", type=float, default=BETA_WARPING, help="庞加莱畸变参数")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM 模型名称")
    
    args = parser.parse_args()
    
    run_experiment(
        qa_file=args.qa,
        chunks_file=args.chunks,
        output_file=args.output,
        num_to_retrieve=args.k,
        beta=args.beta,
        model_name=args.model
    )

