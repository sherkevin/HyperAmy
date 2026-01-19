#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成Emotion Embedding训练数据集

格式：
{
  "text": "I am absolutely furious right now!",
  "soft_label": [0.01, 0.02, 0.85, 0.05, ...],  // 28-dim probability vector
  "intensity": 0.85  // Max-Norm, computed as max(soft_label)
}

28种情绪：
Positive (12): admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
Negative (11): anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
Ambiguous/Cognitive (4): confusion, curiosity, realization, surprise
Neutral (1): neutral
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import logging

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

# 28种情绪列表（按顺序，与用户指定的一致）
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

# 确保是28维
assert len(EMOTIONS) == 28, f"情绪列表必须是28维，当前是{len(EMOTIONS)}维"

EMOTION_PROMPT_TEMPLATE = """Analyze the emotional content of the following text and provide a probability distribution over 28 emotions.

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

Text: {text}

Please provide a probability distribution (soft label) as a JSON array of 28 numbers, where each number represents the probability of that emotion. The probabilities should sum to 1.0 or close to 1.0.

Output format (JSON only, no explanation):
{{
  "soft_label": [0.01, 0.02, 0.85, 0.05, ...]
}}

Return only the JSON object, nothing else."""


def extract_emotion_soft_label(text: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    使用LLM提取文本的28维情绪向量
    
    Args:
        text: 输入文本
        max_retries: 最大重试次数
    
    Returns:
        Dict包含soft_label和intensity，如果失败返回None
    """
    client = CompletionClient()
    prompt = EMOTION_PROMPT_TEMPLATE.format(text=text)
    
    for attempt in range(max_retries):
        try:
            # 使用complete方法，mode="normal"使用chat API
            result = client.complete(
                query=prompt,
                mode="normal",  # 使用chat API
                max_tokens=200,
                temperature=0.3  # 较低温度以保持一致性
            )
            
            # 获取响应文本（ChatResult对象有get_answer_text方法）
            if hasattr(result, 'get_answer_text'):
                response_text = result.get_answer_text().strip()
            elif hasattr(result, 'answer_text'):
                response_text = result.answer_text.strip()
            else:
                response_text = str(result).strip()
            
            # 尝试提取JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # 解析JSON
            result = json.loads(response_text)
            soft_label = result.get('soft_label', [])
            
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


def load_texts_from_qa_file(qa_file: Path) -> List[str]:
    """从QA文件中提取文本"""
    texts = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    for item in qa_data:
        # 添加chunk_text
        if 'chunk_text' in item and item['chunk_text']:
            texts.append(item['chunk_text'].strip())
        
        # 添加question（可选，如果需要的话）
        # if 'question' in item and item['question']:
        #     texts.append(item['question'].strip())
        
        # 添加answer（可选）
        # if 'answer' in item and item['answer']:
        #     texts.append(item['answer'].strip())
    
    return texts


def load_texts_from_training_file(training_file: Path) -> List[str]:
    """从训练文件中提取文本"""
    texts = []
    with open(training_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'input' in data and data['input']:
                    texts.append(data['input'].strip())
    return texts


def generate_training_data(
    input_file: Path,
    output_file: Path,
    max_samples: int = None,
    use_qa: bool = True,
    batch_size: int = 1
):
    """
    生成训练数据
    
    Args:
        input_file: 输入文件路径（QA或训练数据）
        output_file: 输出文件路径
        max_samples: 最大样本数（None表示全部）
        use_qa: 是否使用QA文件格式
        batch_size: 批处理大小（当前为1，因为LLM调用）
    """
    logger.info(f"加载文本数据: {input_file}")
    
    # 加载文本
    if use_qa:
        texts = load_texts_from_qa_file(input_file)
    else:
        texts = load_texts_from_training_file(input_file)
    
    if max_samples:
        texts = texts[:max_samples]
    
    logger.info(f"共加载 {len(texts)} 个文本")
    
    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理每个文本
    success_count = 0
    failed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, text in enumerate(tqdm(texts, desc="提取情绪向量")):
            if not text or len(text.strip()) < 10:  # 跳过太短的文本
                logger.debug(f"跳过文本 {idx}: 太短")
                continue
            
            # 提取情绪向量
            result = extract_emotion_soft_label(text)
            
            if result:
                # 创建训练样本
                sample = {
                    "text": text,
                    "soft_label": result["soft_label"],
                    "intensity": result["intensity"]
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                success_count += 1
            else:
                logger.warning(f"文本 {idx} 提取失败: {text[:50]}...")
                failed_count += 1
    
    logger.info(f"✅ 完成！成功: {success_count}, 失败: {failed_count}")
    logger.info(f"输出文件: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成Emotion Embedding训练数据集")
    parser.add_argument(
        "--input",
        type=str,
        default="data/public_benchmark/monte_cristo_qa_full.json",
        help="输入文件路径（QA或训练数据）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/emotion_training_data.jsonl",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数（None表示全部）"
    )
    parser.add_argument(
        "--use-qa",
        action="store_true",
        default=True,
        help="使用QA文件格式"
    )
    parser.add_argument(
        "--use-training",
        action="store_true",
        default=False,
        help="使用训练文件格式"
    )
    
    args = parser.parse_args()
    
    input_file = Path(project_root) / args.input
    output_file = Path(project_root) / args.output
    
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        sys.exit(1)
    
    use_qa = args.use_qa and not args.use_training
    
    logger.info("=" * 80)
    logger.info("生成Emotion Embedding训练数据集")
    logger.info("=" * 80)
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"最大样本数: {args.max_samples or '全部'}")
    logger.info(f"使用格式: {'QA' if use_qa else 'Training'}")
    logger.info("=" * 80)
    
    generate_training_data(
        input_file=input_file,
        output_file=output_file,
        max_samples=args.max_samples,
        use_qa=use_qa
    )

