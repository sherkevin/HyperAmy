#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成实体粒度情绪数据集

新格式：
{
  "text": "The cat ate the cat food.",
  "targets": [
    {
      "span_text": "cat",
      "char_start": 4,
      "char_end": 7,
      "soft_label": [0.01, 0.02, 0.85, 0.05, ...],  // 28-dim probability vector
      "intensity": 0.85  // Max-Norm of soft_label
    }
  ]
}

处理流程：
1. 从文本中提取实体（使用spaCy获取精确字符位置）
2. 为每个实体提取28维soft_label（使用LLM）
3. 计算intensity（max-norm）
4. 生成新格式的JSON
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.completion_client import CompletionClient
from llm.config import API_KEY, API_URL_CHAT, DEFAULT_MODEL

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 28种情绪列表（按顺序，与generate_emotion_training_data.py一致）
EMOTIONS = [
    # Positive (12)
    "admiration", "amusement", "approval", "caring", "desire",
    "excitement", "gratitude", "joy", "love", "optimism",
    "pride", "relief",
    
    # Negative (11)
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse",
    "sadness",
    
    # Ambiguous / Cognitive (4)
    "confusion", "curiosity", "realization", "surprise",
    
    # Neutral (1)
    "neutral"
]

assert len(EMOTIONS) == 28, f"情绪列表必须是28维，当前是{len(EMOTIONS)}维"

EMOTION_PROMPT_TEMPLATE = """Analyze the emotional content of the following text span and provide a probability distribution over 28 emotions.

The 28 emotions are (in order):
1. admiration
2. amusement
3. approval
4. caring
5. desire
6. excitement
7. gratitude
8. joy
9. love
10. optimism
11. pride
12. relief
13. anger
14. annoyance
15. disappointment
16. disapproval
17. disgust
18. embarrassment
19. fear
20. grief
21. nervousness
22. remorse
23. sadness
24. confusion
25. curiosity
26. realization
27. surprise
28. neutral

Text span: {text}

Context (full text): {context}

Please provide a probability distribution (soft label) as a JSON array of 28 numbers, where each number represents the probability of that emotion. The probabilities should sum to 1.0 or close to 1.0.

Output format (JSON only, no explanation):
{{
  "soft_label": [0.01, 0.02, 0.85, 0.05, ...]
}}

Return only the JSON object, nothing else."""


def extract_entities_with_positions(text: str) -> List[Dict[str, Any]]:
    """
    使用spaCy提取实体及其字符位置
    
    Args:
        text: 输入文本
        
    Returns:
        实体列表，每个实体包含：
        - span_text: 实体文本
        - char_start: 开始字符位置
        - char_end: 结束字符位置
        - entity_type: 实体类型（可选）
    """
    try:
        import spacy
        from utils.ner_lightweight import LightweightNER
    except ImportError:
        logger.error("spaCy or LightweightNER not available. Please install: pip install spacy")
        # 使用备选方案：基于规则提取
        return extract_entities_with_positions_fallback(text)
    
    try:
        # 使用spaCy直接获取字符位置
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        entities = []
        seen_positions = set()  # 避免重复实体
        
        relevant_entity_types = {
            'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
            'WORK_OF_ART', 'LAW', 'LANGUAGE', 'NORP', 'FAC'
        }
        
        for ent in doc.ents:
            if ent.label_ in relevant_entity_types:
                char_start = ent.start_char
                char_end = ent.end_char
                span_text = ent.text.strip()
                
                # 跳过太短的实体
                if len(span_text) < 2:
                    continue
                
                # 检查是否已存在（基于位置去重）
                position_key = (char_start, char_end)
                if position_key in seen_positions:
                    continue
                seen_positions.add(position_key)
                
                entities.append({
                    'span_text': span_text,
                    'char_start': char_start,
                    'char_end': char_end,
                    'entity_type': ent.label_
                })
        
        # 如果没有找到实体，使用名词短语补充
        if len(entities) == 0:
            for chunk in doc.noun_chunks:
                char_start = chunk.start_char
                char_end = chunk.end_char
                span_text = chunk.text.strip()
                
                if len(span_text) >= 2:
                    position_key = (char_start, char_end)
                    if position_key not in seen_positions:
                        seen_positions.add(position_key)
                        entities.append({
                            'span_text': span_text,
                            'char_start': char_start,
                            'char_end': char_end,
                            'entity_type': 'NOUN_PHRASE'
                        })
        
        # 按出现顺序排序
        entities.sort(key=lambda x: x['char_start'])
        
        return entities
        
    except Exception as e:
        logger.warning(f"spaCy实体提取失败: {e}，使用备选方案")
        return extract_entities_with_positions_fallback(text)


def extract_entities_with_positions_fallback(text: str) -> List[Dict[str, Any]]:
    """
    备选方案：基于规则提取实体（不使用spaCy时）
    """
    import re
    
    entities = []
    seen_positions = set()
    
    # 匹配大写开头的多词短语（可能的人名、地名等）
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    for match in re.finditer(pattern, text):
        char_start = match.start()
        char_end = match.end()
        span_text = match.group().strip()
        
        if len(span_text) >= 2:
            position_key = (char_start, char_end)
            if position_key not in seen_positions:
                seen_positions.add(position_key)
                entities.append({
                    'span_text': span_text,
                    'char_start': char_start,
                    'char_end': char_end,
                    'entity_type': 'CAPITALIZED_PHRASE'
                })
    
    entities.sort(key=lambda x: x['char_start'])
    return entities


def extract_emotion_soft_label(span_text: str, context: str = "", max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    使用LLM提取实体span的28维情绪向量
    
    Args:
        span_text: 实体文本span
        context: 完整上下文文本
        max_retries: 最大重试次数
    
    Returns:
        Dict包含soft_label和intensity，如果失败返回None
    """
    client = CompletionClient()
    prompt = EMOTION_PROMPT_TEMPLATE.format(text=span_text, context=context[:500])  # 限制上下文长度
    
    for attempt in range(max_retries):
        try:
            result = client.complete(
                query=prompt,
                mode="normal",
                max_tokens=200,
                temperature=0.3
            )
            
            # 获取响应文本
            if hasattr(result, 'get_answer_text'):
                response_text = result.get_answer_text().strip()
            elif hasattr(result, 'answer_text'):
                response_text = result.answer_text.strip()
            else:
                response_text = str(result).strip()
            
            # 提取JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # 解析JSON
            result_json = json.loads(response_text)
            soft_label = result_json.get('soft_label', [])
            
            # 验证维度
            if len(soft_label) != 28:
                logger.warning(f"维度不匹配: {len(soft_label)} != 28，尝试修复...")
                if len(soft_label) < 28:
                    soft_label.extend([0.0] * (28 - len(soft_label)))
                else:
                    soft_label = soft_label[:28]
            
            # 归一化确保和为1.0
            total = sum(soft_label)
            if total > 0:
                soft_label = [s / total for s in soft_label]
            else:
                logger.warning(f"所有概率为0，使用均匀分布")
                soft_label = [1.0 / 28.0] * 28
            
            # 计算intensity (Max-Norm)
            intensity = max(soft_label)
            
            return {
                "soft_label": soft_label,
                "intensity": intensity
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            else:
                logger.error(f"无法解析JSON: {response_text[:200]}")
                return None
        except Exception as e:
            logger.warning(f"提取失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return None
    
    return None


def process_single_text(
    text: str,
    text_idx: int,
    max_entities_per_text: int = 10,
    enable_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    处理单个文本，提取实体并生成情绪标签
    
    Args:
        text: 输入文本
        text_idx: 文本索引
        max_entities_per_text: 每个文本最大实体数
        enable_cache: 是否启用缓存
        cache_dir: 缓存目录
    
    Returns:
        新格式的数据字典，如果失败返回None
    """
    if not text or len(text.strip()) < 10:
        return None
    
    # 提取实体及其位置
    entities = extract_entities_with_positions(text)
    
    if len(entities) == 0:
        logger.debug(f"文本 {text_idx} 未找到实体")
        return None
    
    # 限制实体数量
    entities = entities[:max_entities_per_text]
    
    # 为每个实体提取情绪标签
    targets = []
    for entity in entities:
        span_text = entity['span_text']
        
        # 检查缓存
        cache_key = None
        cached_result = None
        if enable_cache and cache_dir:
            cache_key = hashlib.md5(f"{span_text}:{text[:100]}".encode()).hexdigest()
            cache_file = cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_result = json.load(f)
                except Exception as e:
                    logger.debug(f"读取缓存失败: {e}")
        
        if cached_result:
            emotion_result = cached_result
        else:
            # 提取情绪标签
            emotion_result = extract_emotion_soft_label(span_text, context=text)
            
            # 保存到缓存
            if emotion_result and enable_cache and cache_dir and cache_key:
                cache_file = cache_dir / f"{cache_key}.json"
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(emotion_result, f)
                except Exception as e:
                    logger.debug(f"保存缓存失败: {e}")
        
        if emotion_result:
            targets.append({
                'span_text': span_text,
                'char_start': entity['char_start'],
                'char_end': entity['char_end'],
                'soft_label': emotion_result['soft_label'],
                'intensity': emotion_result['intensity']
            })
    
    if len(targets) == 0:
        return None
    
    return {
        'text': text,
        'targets': targets
    }


def generate_entity_granularity_dataset(
    input_file: Path,
    output_file: Path,
    max_samples: Optional[int] = None,
    max_entities_per_text: int = 10,
    max_workers: int = 10,
    enable_cache: bool = True
):
    """
    生成实体粒度数据集
    
    Args:
        input_file: 输入文件路径（JSONL格式）
        output_file: 输出文件路径
        max_samples: 最大样本数（None表示全部）
        max_entities_per_text: 每个文本最大实体数
        max_workers: 并发处理线程数
        enable_cache: 是否启用缓存
    """
    logger.info(f"加载文本数据: {input_file}")
    
    # 加载文本
    texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text = data.get('input') or data.get('text') or data.get('content', '')
                if text and len(text.strip()) >= 10:
                    texts.append(text.strip())
    
    if max_samples:
        texts = texts[:max_samples]
    
    logger.info(f"共加载 {len(texts)} 个文本")
    
    # 创建输出目录和缓存目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_file.parent / ".cache_emotion_entities"
    if enable_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否有进度文件（断点续传）
    progress_file = output_file.parent / f"{output_file.stem}_progress.json"
    processed_indices = set()
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                processed_indices = set(progress_data.get('processed_indices', []))
            logger.info(f"从进度文件恢复，已处理 {len(processed_indices)} 个样本")
        except Exception as e:
            logger.warning(f"读取进度文件失败: {e}")
    
    # 处理文本（并发）
    success_count = 0
    failed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务（跳过已处理的）
            future_to_idx = {
                executor.submit(
                    process_single_text,
                    text,
                    idx,
                    max_entities_per_text,
                    enable_cache,
                    cache_dir
                ): idx
                for idx, text in enumerate(texts)
                if idx not in processed_indices
            }
            
            # 处理结果
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="处理文本"):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()  # 立即写入
                        success_count += 1
                    else:
                        failed_count += 1
                    
                    # 更新进度
                    processed_indices.add(idx)
                    with open(progress_file, 'w', encoding='utf-8') as pf:
                        json.dump({'processed_indices': list(processed_indices)}, pf)
                        
                except Exception as e:
                    logger.error(f"处理文本 {idx} 失败: {e}")
                    failed_count += 1
    
    logger.info(f"✅ 完成！成功: {success_count}, 失败: {failed_count}")
    logger.info(f"输出文件: {output_file}")
    
    # 清理进度文件
    if progress_file.exists():
        progress_file.unlink()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成实体粒度情绪数据集")
    parser.add_argument(
        "--input",
        type=str,
        default="data/training/monte_cristo_train_full.jsonl",
        help="输入文件路径（JSONL格式）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/entity_granularity/entity_granularity_monte_cristo.jsonl",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数（None表示全部）"
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=10,
        help="每个文本最大实体数"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="并发处理线程数"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用缓存"
    )
    
    args = parser.parse_args()
    
    input_file = Path(project_root) / args.input
    output_file = Path(project_root) / args.output
    
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("生成实体粒度情绪数据集")
    logger.info("=" * 80)
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"最大样本数: {args.max_samples or '全部'}")
    logger.info(f"每个文本最大实体数: {args.max_entities}")
    logger.info(f"并发线程数: {args.max_workers}")
    logger.info(f"缓存: {'禁用' if args.no_cache else '启用'}")
    logger.info("=" * 80)
    
    generate_entity_granularity_dataset(
        input_file=input_file,
        output_file=output_file,
        max_samples=args.max_samples,
        max_entities_per_text=args.max_entities,
        max_workers=args.max_workers,
        enable_cache=not args.no_cache
    )

