#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新运行失败的问题（并发优化版本）

用法:
    python src/retry_failed_questions_parallel.py --input results/experiment_full.json --output results/experiment_full_retried.json --workers 3
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.run_experiment import (
    load_qa_data, load_amygdala_chunks, build_faiss_index,
    retrieve_baseline, retrieve_hyperamy, generate_answer_with_llm
)
from sentence_transformers import SentenceTransformer
from llm import create_client
from llm.config import DEFAULT_MODEL, BETA_WARPING
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 线程锁，用于保护文件写入
file_lock = threading.Lock()


def is_answer_valid(answer: str) -> bool:
    """检查答案是否有效"""
    if not answer or not isinstance(answer, str):
        return False
    invalid_keywords = ['出错', 'Error', 'error', 'HTTPSConnectionPool', 'ConnectionError', 'Timeout']
    return not any(keyword in answer for keyword in invalid_keywords)


def process_single_question(
    idx: int,
    qa_pair: Dict,
    chunk_dict: Dict,
    embedding_model: SentenceTransformer,
    faiss_index,
    chunks: List[Dict],
    chunk_id_to_idx: Dict,
    llm_client,
    model_name: str,
    beta: float,
    num_to_retrieve: int = 3
) -> Tuple[int, Dict]:
    """
    处理单个问题（生成3组答案）
    
    Returns:
        (idx, result_dict)
    """
    question = qa_pair['question']
    gold_answer = qa_pair['answer']
    gold_chunk_id = qa_pair['chunk_id']
    
    logger.info(f"Processing question {idx+1}: {question[:50]}...")
    
    try:
        # 获取查询向量
        query_vec = embedding_model.encode(question, convert_to_numpy=True)
        
        # Group A: Oracle（直接使用正确答案的 Chunk）
        oracle_chunk = chunk_dict.get(gold_chunk_id)
        if oracle_chunk:
            oracle_context = [oracle_chunk['text']]
        else:
            oracle_context = ["[未找到对应的 chunk]"]
        
        # Group B: Baseline（标准 Cosine 检索）
        baseline_docs, baseline_scores = retrieve_baseline(
            query_vec, faiss_index, chunks, chunk_id_to_idx, num_to_retrieve
        )
        
        # Group C: HyperAmy（庞加莱畸变公式检索）
        hyperamy_docs, hyperamy_scores = retrieve_hyperamy(
            query_vec, chunks, num_to_retrieve, beta
        )
        
        # 并发生成3组答案
        def generate_oracle():
            return generate_answer_with_llm(question, oracle_context, llm_client, model_name, max_retries=5)
        
        def generate_baseline():
            return generate_answer_with_llm(question, baseline_docs, llm_client, model_name, max_retries=5)
        
        def generate_hyperamy():
            return generate_answer_with_llm(question, hyperamy_docs, llm_client, model_name, max_retries=5)
        
        # 使用线程池并发生成答案
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_oracle = executor.submit(generate_oracle)
            future_baseline = executor.submit(generate_baseline)
            future_hyperamy = executor.submit(generate_hyperamy)
            
            oracle_answer = future_oracle.result()
            baseline_answer = future_baseline.result()
            hyperamy_answer = future_hyperamy.result()
        
        # 检查是否检索到正确答案的 chunk
        baseline_hit = gold_chunk_id in [chunks[chunk_id_to_idx.get(cid, -1)]['chunk_id'] 
                                         for cid in range(len(chunks)) 
                                         if chunk_id_to_idx.get(cid, -1) >= 0][:num_to_retrieve]
        hyperamy_hit = gold_chunk_id in [chunk['chunk_id'] for chunk in chunks if chunk['text'] in hyperamy_docs]
        
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
                'hit': baseline_hit
            },
            'hyperamy': {
                'context': hyperamy_docs,
                'scores': hyperamy_scores.tolist(),
                'answer': hyperamy_answer,
                'hit': hyperamy_hit
            }
        }
        
        return idx, result
        
    except Exception as e:
        logger.error(f"Error processing question {idx+1}: {e}")
        import traceback
        traceback.print_exc()
        # 返回错误结果
        return idx, {
            'question': question,
            'gold_answer': gold_answer,
            'gold_chunk_id': gold_chunk_id,
            'error': str(e)
        }


def save_results_safely(results: List[Dict], output_file: str):
    """线程安全地保存结果"""
    with file_lock:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def retry_failed_questions_parallel(
    input_file: str,
    output_file: str,
    qa_file: str = "data/benchmarks/instinct_qa.json",
    chunks_file: str = "data/processed/got_amygdala.jsonl",
    model_name: str = DEFAULT_MODEL,
    max_workers: int = 3
):
    """
    重新运行失败的问题（并发版本）
    
    Args:
        input_file: 原始实验结果文件
        output_file: 输出文件路径
        qa_file: QA数据文件
        chunks_file: 分块数据文件
        model_name: LLM模型名称
        max_workers: 并发处理的问题数量
    """
    # 1. 加载原始结果
    logger.info(f"Loading original results from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_results = json.load(f)
    logger.info(f"Loaded {len(original_results)} results")
    
    # 2. 找出失败的问题
    failed_indices = []
    for i, result in enumerate(original_results):
        oracle_ok = is_answer_valid(result.get('oracle', {}).get('answer', ''))
        baseline_ok = is_answer_valid(result.get('baseline', {}).get('answer', ''))
        hyperamy_ok = is_answer_valid(result.get('hyperamy', {}).get('answer', ''))
        
        if not (oracle_ok and baseline_ok and hyperamy_ok):
            failed_indices.append(i)
    
    logger.info(f"Found {len(failed_indices)} failed questions: {failed_indices}")
    
    if len(failed_indices) == 0:
        logger.info("No failed questions found. All results are valid!")
        return
    
    # 3. 加载数据
    logger.info(f"Loading QA data from {qa_file}...")
    qa_pairs = load_qa_data(qa_file)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    logger.info(f"Loading chunks from {chunks_file}...")
    chunks = load_amygdala_chunks(chunks_file)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # 4. 构建映射和索引
    chunk_dict = {chunk['chunk_id']: chunk for chunk in chunks}
    
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logger.info("Building FAISS index...")
    faiss_index, chunk_id_to_idx = build_faiss_index(chunks, embedding_model)
    
    # 5. 初始化LLM客户端（每个线程共享同一个客户端）
    logger.info(f"Initializing LLM client with model: {model_name}...")
    llm_client = create_client(model_name=model_name, mode="normal")
    
    # 6. 准备结果列表（复制原始结果）
    updated_results = original_results.copy()
    
    # 7. 并发处理失败的问题
    logger.info(f"Retrying {len(failed_indices)} failed questions with {max_workers} workers...")
    
    # 准备任务参数
    tasks = []
    for idx in failed_indices:
        if idx < len(qa_pairs):
            tasks.append((idx, qa_pairs[idx]))
    
    # 使用线程池并发处理
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(
                process_single_question,
                idx, qa_pair, chunk_dict, embedding_model,
                faiss_index, chunks, chunk_id_to_idx,
                llm_client, model_name, BETA_WARPING
            ): idx
            for idx, qa_pair in tasks
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc="Retrying failed questions") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_idx, result = future.result()
                    updated_results[result_idx] = result
                    completed_count += 1
                    
                    # 每完成5个问题保存一次
                    if completed_count % 5 == 0:
                        save_results_safely(updated_results, output_file)
                        logger.info(f"Intermediate results saved after {completed_count} questions")
                    
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing question {idx}: {e}")
                    pbar.update(1)
    
    # 8. 最终保存
    save_results_safely(updated_results, output_file)
    logger.info(f"Retry completed. Results saved to {output_file}")
    
    # 9. 统计
    valid_count = sum(1 for r in updated_results 
                     if is_answer_valid(r.get('oracle', {}).get('answer', '')) and
                        is_answer_valid(r.get('baseline', {}).get('answer', '')) and
                        is_answer_valid(r.get('hyperamy', {}).get('answer', '')))
    
    logger.info(f"Valid results: {valid_count} / {len(updated_results)} ({100*valid_count/len(updated_results):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retry failed questions in experiment results (parallel version)")
    parser.add_argument("--input", type=str, default="results/experiment_full.json",
                       help="Input experiment results file")
    parser.add_argument("--output", type=str, default="results/experiment_full_retried.json",
                       help="Output file path")
    parser.add_argument("--qa", type=str, default="data/benchmarks/instinct_qa.json",
                       help="QA data file")
    parser.add_argument("--chunks", type=str, default="data/processed/got_amygdala.jsonl",
                       help="Chunks data file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="LLM model name")
    parser.add_argument("--workers", type=int, default=3,
                       help="Number of parallel workers (default: 3)")
    
    args = parser.parse_args()
    
    retry_failed_questions_parallel(
        input_file=args.input,
        output_file=args.output,
        qa_file=args.qa,
        chunks_file=args.chunks,
        model_name=args.model,
        max_workers=args.workers
    )

