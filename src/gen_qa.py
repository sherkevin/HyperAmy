# -*- coding: utf-8 -*-
"""
GoT 实验本能测试题生成模块

功能：
1. 加载高质量块（mass > 0.8）
2. 使用 GPT-4o 验证并生成需要"危机感知"的问题
3. 生成 50 个高质量 (Question, Answer, Chunk_ID) 三元组
"""

import json
import os
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import create_client
from llm.config import MASS_THRESHOLD, DEFAULT_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_amygdala_chunks(file_path: str) -> List[Dict]:
    """
    加载带杏仁核特征的分块数据
    
    Args:
        file_path: got_amygdala.jsonl 文件路径
        
    Returns:
        List[Dict]: 分块列表
    """
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def get_top_k_chunks_by_mass(chunks: List[Dict], top_k: int = 50) -> List[Dict]:
    """
    按质量（mass）排序，返回 Top-K 块（相对阈值策略）
    
    Args:
        chunks: 分块列表
        top_k: 要返回的块数量
        
    Returns:
        List[Dict]: Top-K 高质量分块列表（按 mass 降序排列）
    """
    # 按 mass 降序排序
    sorted_chunks = sorted(chunks, key=lambda x: x.get('mass', 0.0), reverse=True)
    top_chunks = sorted_chunks[:top_k]
    
    if top_chunks:
        logger.info(f"Selected top {len(top_chunks)} chunks by mass from {len(chunks)} total chunks")
        logger.info(f"  Mass range: {top_chunks[-1].get('mass', 0.0):.3f} - {top_chunks[0].get('mass', 0.0):.3f}")
    else:
        logger.warning("No chunks found!")
    
    return top_chunks


def generate_question_for_chunk(chunk: Dict, client, max_retries: int = 3) -> Optional[Dict]:
    """
    为单个块生成问题
    
    Args:
        chunk: 分块字典，包含 text, chunk_id, mass 等
        client: LLM 客户端
        max_retries: 最大重试次数
        
    Returns:
        Optional[Dict]: (Question, Answer, Chunk_ID) 三元组，如果生成失败返回 None
    """
    text = chunk['text']
    chunk_id = chunk['chunk_id']
    mass = chunk.get('mass', 0.0)
    
    prompt = f"""你是一位专业的文学分析专家。请阅读以下来自《冰与火之歌》的高张力文本片段。

文本片段：
{text}

**任务：** 基于这个文本生成一个"高风险问题"（High-Stakes Question）。

**要求：**
1. 问题必须聚焦于**危险信号、突然的认知、或暗示威胁或情节转折的特定细节**。
2. **避免**通用问题，如"谁在说话？"或"发生了什么？"
3. 答案必须**直接**来源于提供的文本。
4. **关键约束：** 构造一个问题，使得仅通过关键词略读的读者可能会错过它，但对**情感基调**敏感的读者能捕捉到它。

**输出 JSON 格式：**
{{
    "question": "需要危机感知或情感理解才能回答的问题",
    "ground_truth_answer": "基于文本的直接答案",
    "key_evidence_snippet": "文本中支持答案的关键证据片段",
    "requires_emotional_sensitivity": true/false,
    "reasoning": "为什么这个问题需要情感敏感度"
}}

请确保问题足够"刁钻"，需要读者对情感基调、隐含威胁或微妙线索有敏锐感知才能回答。"""

    for attempt in range(max_retries):
        try:
            response = client.get_answer(
                prompt,
                max_tokens=500,
                temperature=0.7,
                mode="normal"
            )
            
            # 尝试解析 JSON
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
            # 检查是否有有效的问题和答案
            question = result.get('question', '').strip()
            answer = result.get('ground_truth_answer', result.get('answer', '')).strip()
            
            if question and answer:
                return {
                    'question': question,
                    'answer': answer,
                    'chunk_id': chunk_id,
                    'chunk_text': text[:200] + '...' if len(text) > 200 else text,  # 保存前200字符作为参考
                    'mass': mass,
                    'key_evidence_snippet': result.get('key_evidence_snippet', ''),
                    'requires_emotional_sensitivity': result.get('requires_emotional_sensitivity', True),
                    'reasoning': result.get('reasoning', '')
                }
            
            # 如果解析失败，尝试直接提取
            if 'question' in response.lower() and ('answer' in response.lower() or 'ground_truth' in response.lower()):
                # 简单提取
                lines = response.split('\n')
                question = None
                answer = None
                for i, line in enumerate(lines):
                    if 'question' in line.lower() and ':' in line:
                        question = line.split(':', 1)[1].strip().strip('"').strip("'")
                    if ('ground_truth_answer' in line.lower() or 'answer' in line.lower()) and ':' in line:
                        answer = line.split(':', 1)[1].strip().strip('"').strip("'")
                
                if question and answer:
                    return {
                        'question': question,
                        'answer': answer,
                        'chunk_id': chunk_id,
                        'chunk_text': text[:200] + '...' if len(text) > 200 else text,
                        'mass': mass,
                        'key_evidence_snippet': '',
                        'requires_emotional_sensitivity': True,
                        'reasoning': 'Extracted from unstructured response'
                    }
            
            logger.warning(f"Failed to extract valid question/answer from response for chunk {chunk_id}, attempt {attempt + 1}")
            
        except Exception as e:
            logger.warning(f"Error generating question for chunk {chunk_id}, attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to generate question for chunk {chunk_id} after {max_retries} attempts")
    
    return None


def generate_instinct_qa(
    input_path: str = "data/processed/got_amygdala.jsonl",
    output_path: str = "data/benchmarks/instinct_qa.json",
    num_questions: int = 50,
    top_k: int = 50,
    model_name: str = DEFAULT_MODEL
):
    """
    生成本能测试题（使用 Top-K 策略）
    
    Args:
        input_path: 输入文件路径（got_amygdala.jsonl）
        output_path: 输出文件路径
        num_questions: 要生成的问题数量
        top_k: 按 mass 排序后选取的 Top-K 块数量
        model_name: LLM 模型名称
    """
    # 1. 加载数据
    logger.info(f"Loading chunks from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}. Please run data_prep.py first.")
    
    chunks = load_amygdala_chunks(input_path)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # 2. 按质量排序，选取 Top-K 块（相对阈值策略）
    logger.info(f"Selecting top {top_k} chunks by mass (relative threshold strategy)...")
    top_chunks = get_top_k_chunks_by_mass(chunks, top_k=top_k)
    
    if len(top_chunks) < num_questions:
        logger.warning(f"Only {len(top_chunks)} top chunks available, but {num_questions} questions requested.")
        logger.warning(f"Will generate questions from all {len(top_chunks)} available chunks.")
        high_quality_chunks = top_chunks
    else:
        high_quality_chunks = top_chunks
    
    # 3. 初始化 LLM 客户端
    logger.info(f"Initializing LLM client with model: {model_name}...")
    client = create_client(model_name=model_name, mode="normal")
    
    # 4. 生成问题（使用并发加速）
    logger.info(f"Generating {num_questions} instinct questions...")
    qa_pairs = []
    
    # 使用线程池并发处理（注意：API可能有速率限制，所以限制并发数）
    max_workers = 3  # 限制并发数，避免API限流
    chunks_to_process = high_quality_chunks[:min(num_questions * 2, len(high_quality_chunks))]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_chunk = {
            executor.submit(generate_question_for_chunk, chunk, client): chunk 
            for chunk in chunks_to_process
        }
        
        # 使用tqdm显示进度
        with tqdm(total=min(num_questions, len(chunks_to_process)), desc="Generating questions") as pbar:
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    qa_pair = future.result()
                    if qa_pair:
                        qa_pairs.append(qa_pair)
                        pbar.update(1)
                        if len(qa_pairs) >= num_questions:
                            # 取消剩余任务
                            for f in future_to_chunk:
                                f.cancel()
                            break
                except Exception as e:
                    logger.warning(f"Failed to generate question for chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                    pbar.update(1)
    
    if len(qa_pairs) < num_questions:
        logger.warning(f"Only generated {len(qa_pairs)} questions, less than requested {num_questions}")
    
    # 5. 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Generated {len(qa_pairs)} instinct questions and saved to {output_path}")
    
    # 6. 统计信息
    if qa_pairs:
        avg_mass = sum(qa['mass'] for qa in qa_pairs) / len(qa_pairs)
        logger.info(f"Statistics:")
        logger.info(f"  Total questions: {len(qa_pairs)}")
        logger.info(f"  Average mass: {avg_mass:.3f}")
        logger.info(f"  Min mass: {min(qa['mass'] for qa in qa_pairs):.3f}")
        logger.info(f"  Max mass: {max(qa['mass'] for qa in qa_pairs):.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GoT 实验本能测试题生成")
    parser.add_argument("--input", type=str, default="data/processed/got_amygdala.jsonl", help="输入文件路径")
    parser.add_argument("--output", type=str, default="data/benchmarks/instinct_qa.json", help="输出文件路径")
    parser.add_argument("--num", type=int, default=50, help="要生成的问题数量")
    parser.add_argument("--top-k", type=int, default=50, help="按质量排序后选取的 Top-K 块数量")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM 模型名称")
    
    args = parser.parse_args()
    
    generate_instinct_qa(
        input_path=args.input,
        output_path=args.output,
        num_questions=args.num,
        top_k=args.top_k,
        model_name=args.model
    )

