#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HyperAmy并行运行脚本 - 使用GPU加速

这个脚本可以独立运行HyperAmy索引，与主实验并行执行
"""
import sys
import os
import json
import numpy as np
import logging
from typing import List
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from poincare.storage import HyperAmyStorage
from poincare.retrieval import HyperAmyRetrieval
from particle.particle import ParticleEntity
from point_label.emotion import Emotion
from sentence_transformers import SentenceTransformer

# 设置环境变量
from llm.config import API_KEY
os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['API_KEY'] = API_KEY

# 配置日志
log_file = project_root / 'test_hyperamy_parallel.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("HyperAmy并行运行 - GPU加速版本")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 加载数据集
print("\n【步骤1】加载数据集...")
chunks_file = project_root / "data" / "training" / "monte_cristo_train_full.jsonl"

if not chunks_file.exists():
    logger.error(f"❌ 训练数据文件不存在: {chunks_file}")
    sys.exit(1)

# 加载chunks
chunks = []
with open(chunks_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))
logger.info(f"✅ 加载了 {len(chunks)} 个chunks")

# ========== HyperAmy (纯情绪) - 并行版本 ==========
print("\n【步骤2】初始化 HyperAmy (纯情绪检索，GPU加速)...")
try:
    # 使用poincare的HyperAmyRetrieval
    emotion_extractor = Emotion()
    
    # 使用GPU加速SentenceTransformer（如果可用）
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"✅ 使用GPU加速: {torch.cuda.get_device_name(0)}")
        # 可以选择使用GPU的编号（如果有多个GPU）
        # device = "cuda:0"  # 使用第一个GPU
    else:
        logger.warning("⚠️  GPU不可用，使用CPU")
    
    # SentenceTransformer使用GPU（虽然HyperAmy不直接用它，但初始化时可以使用）
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # 创建存储
    storage_path = project_root / "outputs" / "three_methods_comparison_monte_cristo" / "hyperamy_db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = HyperAmyStorage(persist_path=str(storage_path))
    
    # 维护id->content映射
    id_to_content = {}
    
    # 提取情绪向量并存储点（并发优化 + GPU加速）
    print("   提取情绪向量并存储点（并发处理，GPU加速）...")
    
    def extract_emotion_for_chunk(chunk_data):
        """提取单个chunk的情绪向量（用于并发调用）"""
        chunk_idx, chunk = chunk_data
        # 使用与docs加载相同的字段检查逻辑
        text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
        # 生成chunk_id
        chunk_id = chunk.get('chunk_id') or chunk.get('id') or f'chunk_{chunk_idx}'
        
        # 确保text是字符串且有一定长度
        if isinstance(text, str) and len(text.strip()) > 20:
            try:
                emotion_vector = emotion_extractor.extract(text)
                # 确保emotion_vector是torch.Tensor，并在CPU上（ChromaDB存储需要CPU）
                if isinstance(emotion_vector, np.ndarray):
                    emotion_vector = torch.tensor(emotion_vector, dtype=torch.float32)
                elif not isinstance(emotion_vector, torch.Tensor):
                    emotion_vector = torch.tensor(emotion_vector, dtype=torch.float32)
                # 确保在CPU上（存储需要CPU）
                if emotion_vector.is_cuda:
                    emotion_vector = emotion_vector.cpu()
                
                # 使用chunk_id作为point_id
                point_id = chunk_id
                
                return {
                    'success': True,
                    'chunk_idx': chunk_idx,
                    'chunk_id': chunk_id,
                    'point_id': point_id,
                    'emotion_vector': emotion_vector,
                    'text': text.strip(),
                    'error': None
                }
            except Exception as e:
                logger.warning(f"处理chunk失败 (索引={chunk_idx}, chunk_id={chunk_id}, text_len={len(text) if isinstance(text, str) else 0}): {e}")
                return {
                    'success': False,
                    'chunk_idx': chunk_idx,
                    'chunk_id': chunk_id,
                    'entity_id': None,
                    'emotion_vector': None,
                    'weight': None,
                    'text': None,
                    'error': str(e)
                }
        else:
            return {
                'success': False,
                'chunk_idx': chunk_idx,
                'chunk_id': chunk_id,
                'entity_id': None,
                'emotion_vector': None,
                'weight': None,
                'text': None,
                'error': f"text长度不足或无效 (len={len(text) if isinstance(text, str) else 'N/A'})"
            }
    
    # 使用ThreadPoolExecutor并发处理（增加worker数以利用GPU）
    max_workers = 15  # 增加并发数，充分利用GPU和API并发
    stored_points = 0
    skipped_count = 0
    logger.info(f"使用并发处理，max_workers={max_workers}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_chunk = {
            executor.submit(extract_emotion_for_chunk, (chunk_idx, chunk)): chunk_idx
            for chunk_idx, chunk in enumerate(chunks)
        }
        
        # 收集结果（带进度条）
        results_list = []
        for future in tqdm(as_completed(future_to_chunk), 
                          total=len(chunks), 
                          desc="提取情绪向量（并发+GPU）"):
            try:
                result = future.result()
                results_list.append(result)
            except Exception as e:
                chunk_idx = future_to_chunk[future]
                logger.error(f"并发处理异常 (索引={chunk_idx}): {e}")
                skipped_count += 1
        
        # 按照原始顺序整理结果并存储
        results_list.sort(key=lambda x: x['chunk_idx'])
        
        # 批量存储entities（提高效率）
        entities_batch = []
        for result in results_list:
            if result['success']:
                # 保存id->content映射
                id_to_content[result['entity_id']] = result['text']
                
                # 创建ParticleEntity对象
                entity = ParticleEntity(
                    entity_id=result['entity_id'],
                    entity=f"chunk_{result['chunk_idx']}",  # 实体名称
                    text_id=result['chunk_id'],  # 文本ID
                    emotion_vector=result['emotion_vector'],  # 归一化后的向量
                    weight=result['weight'],  # 原始模长
                    speed=0.0,  # 初始速度
                    temperature=1.0,  # 初始温度
                    born=0.0  # 创建时间
                )
                entities_batch.append(entity)
                
                # 每100个批量写入（减少I/O开销）
                if len(entities_batch) >= 100:
                    for e in entities_batch:
                        storage.upsert_entity(e)
                    stored_points += len(entities_batch)
                    entities_batch = []
        
        # 写入剩余的entities
        if entities_batch:
            for e in entities_batch:
                storage.upsert_entity(e)
            stored_points += len(entities_batch)
    
    print(f"✅ HyperAmy 存储初始化完成（存储了 {stored_points} 个点，跳过了 {skipped_count} 个无效chunks）")
    
    # 保存id_to_content映射到文件（供后续检索使用）
    id_to_content_file = storage_path.parent / "hyperamy_id_to_content.json"
    with open(id_to_content_file, 'w', encoding='utf-8') as f:
        json.dump(id_to_content, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 保存了id->content映射到: {id_to_content_file}")
    
    logger.info("✅ HyperAmy 并行索引完成！")
    
except Exception as e:
    logger.error(f"HyperAmy 模块初始化或运行失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

