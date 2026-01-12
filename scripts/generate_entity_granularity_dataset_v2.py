#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成实体粒度情绪数据集 - 改进版 v2

改进点：
1. 过滤包含"Chapter"的短文本（章节标题）
2. soft_label不进行全局归一化，只限制每个维度在0-1范围（保持模长可变）
3. 支持从QA对生成数据，确保Q和A的实体情绪向量相近
4. 支持从多个数据源加载文本

新格式：
{
  "text": "The cat ate the cat food.",
  "targets": [
    {
      "span_text": "cat",
      "char_start": 4,
      "char_end": 7,
      "soft_label": [0.01, 0.02, 0.85, 0.05, ...],  // 28-dim vector (each in [0,1], NOT normalized)
      "intensity": 0.85  // L2-norm or Max-norm of soft_label
    }
  ]
}
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
import re

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

# 28种情绪列表
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

# 标准情绪提取prompt（用于普通文本）
EMOTION_PROMPT_TEMPLATE = """Analyze the emotional content of the following text span and provide emotion scores over 28 emotions.

The 28 emotions are (in order):
1. admiration, 2. amusement, 3. approval, 4. caring, 5. desire,
6. excitement, 7. gratitude, 8. joy, 9. love, 10. optimism,
11. pride, 12. relief, 13. anger, 14. annoyance, 15. disappointment,
16. disapproval, 17. disgust, 18. embarrassment, 19. fear, 20. grief,
21. nervousness, 22. remorse, 23. sadness, 24. confusion, 25. curiosity,
26. realization, 27. surprise, 28. neutral

Text span: {text}

Context (full text): {context}

Provide emotion scores as a JSON array of 28 numbers, where each number is between 0.0 and 1.0, representing the intensity of that emotion. Each score should be independent (no normalization required).

Output format (JSON only, no explanation):
{{
  "soft_label": [0.01, 0.02, 0.85, 0.05, ...]
}}

Return only the JSON object, nothing else."""

# QA对情绪提取prompt（确保Q和A的情绪向量相近）
EMOTION_PROMPT_QA_TEMPLATE = """Analyze the emotional content of the following text span, which is part of a question-answer pair. Consider that questions and their corresponding answers often share similar emotional undertones.

The 28 emotions are (in order):
1. admiration, 2. amusement, 3. approval, 4. caring, 5. desire,
6. excitement, 7. gratitude, 8. joy, 9. love, 10. optimism,
11. pride, 12. relief, 13. anger, 14. annoyance, 15. disappointment,
16. disapproval, 17. disgust, 18. embarrassment, 19. fear, 20. grief,
21. nervousness, 22. remorse, 23. sadness, 24. confusion, 25. curiosity,
26. realization, 27. surprise, 28. neutral

Text span: {text}

Context (full text): {context}

Related text (from the same QA pair): {related_text}

Provide emotion scores as a JSON array of 28 numbers, where each number is between 0.0 and 1.0, representing the intensity of that emotion. Each score should be independent (no normalization required). When the text span and related text share emotional context, reflect that in the scores.

Output format (JSON only, no explanation):
{{
  "soft_label": [0.01, 0.02, 0.85, 0.05, ...]
}}

Return only the JSON object, nothing else."""


def is_chapter_title(text: str) -> bool:
    """判断是否是章节标题（包含Chapter的短文本）"""
    text_clean = text.strip()
    # 如果文本很短（<50字符）且包含"Chapter"，很可能是章节标题
    if len(text_clean) < 50 and re.search(r'\bChapter\s+\d+', text_clean, re.IGNORECASE):
        return True
    # 如果文本只包含标题和Chapter，也是章节标题
    lines = text_clean.split('\n')
    if len(lines) <= 2:
        for line in lines:
            if re.search(r'\bChapter\s+\d+', line, re.IGNORECASE):
                return True
    return False


def clean_text(text: str) -> Optional[str]:
    """清理文本，移除章节标题等无用内容"""
    text = text.strip()
    if not text or len(text) < 10:
        return None
    
    # 过滤章节标题
    if is_chapter_title(text):
        return None
    
    # 移除开头的章节标题部分（如果存在）
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not re.search(r'\bChapter\s+\d+', line, re.IGNORECASE):
            cleaned_lines.append(line)
        elif len(cleaned_lines) > 0:  # 如果已经有内容，保留后续的Chapter行
            cleaned_lines.append(line)
    
    text_cleaned = '\n'.join(cleaned_lines).strip()
    
    # 如果清理后文本太短，返回None
    if len(text_cleaned) < 10:
        return None
    
    return text_cleaned


def extract_entities_with_positions(text: str) -> List[Dict[str, Any]]:
    """使用spaCy提取实体及其字符位置"""
    try:
        import spacy
    except ImportError:
        logger.error("spaCy not available. Please install: pip install spacy")
        return extract_entities_with_positions_fallback(text)
    
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        entities = []
        seen_positions = set()
        
        relevant_entity_types = {
            'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
            'WORK_OF_ART', 'LAW', 'LANGUAGE', 'NORP', 'FAC'
        }
        
        for ent in doc.ents:
            if ent.label_ in relevant_entity_types:
                char_start = ent.start_char
                char_end = ent.end_char
                span_text = ent.text.strip()
                
                if len(span_text) < 2:
                    continue
                
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
        
        entities.sort(key=lambda x: x['char_start'])
        return entities
        
    except Exception as e:
        logger.warning(f"spaCy实体提取失败: {e}，使用备选方案")
        return extract_entities_with_positions_fallback(text)


def extract_entities_with_positions_fallback(text: str) -> List[Dict[str, Any]]:
    """备选方案：基于规则提取实体"""
    entities = []
    seen_positions = set()
    
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


def extract_emotion_soft_label(
    span_text: str, 
    context: str = "", 
    related_text: str = "",
    is_qa_pair: bool = False,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    使用LLM提取实体span的28维情绪向量
    
    Args:
        span_text: 实体文本span
        context: 完整上下文文本
        related_text: 相关文本（用于QA对）
        is_qa_pair: 是否是QA对的一部分
        max_retries: 最大重试次数
    
    Returns:
        Dict包含soft_label和intensity，如果失败返回None
    """
    client = CompletionClient()
    
    if is_qa_pair and related_text:
        prompt = EMOTION_PROMPT_QA_TEMPLATE.format(
            text=span_text, 
            context=context[:500],
            related_text=related_text[:200]
        )
    else:
        prompt = EMOTION_PROMPT_TEMPLATE.format(
            text=span_text, 
            context=context[:500]
        )
    
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
            
            # 限制每个维度在[0, 1]范围，但不做全局归一化
            soft_label = [max(0.0, min(1.0, float(s))) for s in soft_label]
            
            # 计算intensity (L2-norm，更合理)
            intensity = float(np.linalg.norm(soft_label))
            
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
    cache_dir: Optional[Path] = None,
    related_text: str = "",
    is_qa_pair: bool = False
) -> Optional[Dict[str, Any]]:
    """处理单个文本，提取实体并生成情绪标签"""
    # 清理文本
    text = clean_text(text)
    if not text:
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
            cache_key_str = f"{span_text}:{text[:100]}:{related_text[:50]}" if is_qa_pair else f"{span_text}:{text[:100]}"
            cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
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
            emotion_result = extract_emotion_soft_label(
                span_text, 
                context=text,
                related_text=related_text,
                is_qa_pair=is_qa_pair
            )
            
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


def load_texts_from_training_file(training_file: Path) -> List[str]:
    """从训练文件加载文本"""
    texts = []
    with open(training_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text = data.get('input') or data.get('text') or data.get('content', '')
                if text and len(text.strip()) >= 10:
                    texts.append(text.strip())
    return texts


def load_qa_pairs(qa_file: Path) -> List[Dict[str, Any]]:
    """从QA文件加载QA对"""
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    return qa_pairs


def generate_entity_granularity_dataset(
    training_files: List[Path],
    qa_files: List[Path],
    output_file: Path,
    max_samples_per_source: Optional[int] = None,
    max_entities_per_text: int = 10,
    max_workers: int = 10,
    enable_cache: bool = True,
    qa_ratio: float = 0.3  # QA对在总数据中的比例
):
    """
    生成实体粒度数据集（支持多数据源）
    
    Args:
        training_files: 训练文本文件列表（JSONL格式）
        qa_files: QA文件列表（JSON格式）
        output_file: 输出文件路径
        max_samples_per_source: 每个数据源最大样本数
        max_entities_per_text: 每个文本最大实体数
        max_workers: 并发处理线程数
        enable_cache: 是否启用缓存
        qa_ratio: QA对在总数据中的目标比例
    """
    logger.info("=" * 80)
    logger.info("生成实体粒度情绪数据集 (改进版 v2)")
    logger.info("=" * 80)
    
    # 加载所有文本数据
    all_texts = []
    
    # 从训练文件加载
    for training_file in training_files:
        if training_file.exists():
            logger.info(f"加载训练文本: {training_file}")
            texts = load_texts_from_training_file(training_file)
            if max_samples_per_source:
                texts = texts[:max_samples_per_source]
            all_texts.extend([(text, False, "") for text in texts])
            logger.info(f"  加载了 {len(texts)} 个文本")
    
    # 从QA文件加载
    qa_texts = []
    for qa_file in qa_files:
        if qa_file.exists():
            logger.info(f"加载QA对: {qa_file}")
            qa_pairs = load_qa_pairs(qa_file)
            for qa in qa_pairs:
                question = qa.get('question', '')
                answer = qa.get('answer', '') or qa.get('chunk_text', '')
                if question and len(question.strip()) >= 10:
                    qa_texts.append((question, True, answer))
                if answer and len(answer.strip()) >= 10:
                    qa_texts.append((answer, True, question))
            logger.info(f"  加载了 {len(qa_pairs)} 个QA对，生成 {len(qa_texts)} 个文本")
    
    # 根据qa_ratio调整QA文本数量
    if qa_texts and all_texts:
        target_qa_count = int(len(all_texts) * qa_ratio / (1 - qa_ratio))
        if len(qa_texts) > target_qa_count:
            qa_texts = qa_texts[:target_qa_count]
            logger.info(f"  调整QA文本数量为 {len(qa_texts)} (目标比例: {qa_ratio})")
    
    all_texts.extend(qa_texts)
    
    logger.info(f"共加载 {len(all_texts)} 个文本（其中 {sum(1 for _, is_qa, _ in all_texts if is_qa)} 个来自QA对）")
    
    # 创建输出目录和缓存目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_file.parent / ".cache_emotion_entities"
    if enable_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查进度文件
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
            # 提交所有任务
            future_to_idx = {
                executor.submit(
                    process_single_text,
                    text,
                    idx,
                    max_entities_per_text,
                    enable_cache,
                    cache_dir,
                    related_text,
                    is_qa
                ): idx
                for idx, (text, is_qa, related_text) in enumerate(all_texts)
                if idx not in processed_indices
            }
            
            # 处理结果
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="处理文本"):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
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
    
    parser = argparse.ArgumentParser(description="生成实体粒度情绪数据集 (改进版 v2)")
    parser.add_argument(
        "--training-files",
        type=str,
        nargs='+',
        default=["data/training/monte_cristo_train_full.jsonl"],
        help="训练文本文件路径（JSONL格式，可多个）"
    )
    parser.add_argument(
        "--qa-files",
        type=str,
        nargs='+',
        default=["data/public_benchmark/monte_cristo_qa_full.json"],
        help="QA文件路径（JSON格式，可多个）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/entity_granularity/entity_granularity_v2.jsonl",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=None,
        help="每个数据源最大样本数"
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
        "--qa-ratio",
        type=float,
        default=0.3,
        help="QA对在总数据中的目标比例"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用缓存"
    )
    
    args = parser.parse_args()
    
    training_files = [Path(project_root) / f for f in args.training_files]
    qa_files = [Path(project_root) / f for f in args.qa_files]
    output_file = Path(project_root) / args.output
    
    logger.info(f"训练文件: {[str(f) for f in training_files]}")
    logger.info(f"QA文件: {[str(f) for f in qa_files]}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"每个数据源最大样本数: {args.max_samples_per_source or '全部'}")
    logger.info(f"每个文本最大实体数: {args.max_entities}")
    logger.info(f"并发线程数: {args.max_workers}")
    logger.info(f"QA对比例: {args.qa_ratio}")
    logger.info(f"缓存: {'禁用' if args.no_cache else '启用'}")
    logger.info("=" * 80)
    
    generate_entity_granularity_dataset(
        training_files=training_files,
        qa_files=qa_files,
        output_file=output_file,
        max_samples_per_source=args.max_samples_per_source,
        max_entities_per_text=args.max_entities,
        max_workers=args.max_workers,
        enable_cache=not args.no_cache,
        qa_ratio=args.qa_ratio
    )
