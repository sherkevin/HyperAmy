#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HyperAmy修复验证脚本 - 小规模测试（10个查询）

用于快速验证HyperAmy检索修复是否有效
"""
import sys
import os
import json
import numpy as np
import logging
import hashlib
from typing import List
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.config import API_KEY, BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
from poincare.storage import HyperAmyStorage
from poincare.retrieval import HyperAmyRetrieval
from particle.particle import ParticleEntity
from point_label.emotion import Emotion
from sentence_transformers import SentenceTransformer

# 设置环境变量
os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['API_KEY'] = API_KEY

# 配置日志
log_file = project_root / 'test_hyperamy_quick_validation.log'
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
print("HyperAmy修复验证 - 小规模测试（10个查询）")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 加载数据集
print("\n【步骤1】加载数据集...")
chunks_file = project_root / "data" / "training" / "monte_cristo_train_full.jsonl"
qa_file = project_root / "data" / "public_benchmark" / "monte_cristo_qa_full.json"

if not chunks_file.exists():
    logger.error(f"❌ 训练数据文件不存在: {chunks_file}")
    sys.exit(1)

if not qa_file.exists():
    logger.error(f"❌ QA数据文件不存在: {qa_file}")
    sys.exit(1)

# 加载chunks
chunks = []
with open(chunks_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))
logger.info(f"✅ 加载了 {len(chunks)} 个chunks")

# 加载QA数据
with open(qa_file, 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)
logger.info(f"✅ 加载了 {len(qa_pairs)} 个QA对")

# 限制为10个查询进行快速测试
MAX_QUERIES = 10
if len(qa_pairs) > MAX_QUERIES:
    logger.info(f"\n⚠️  限制为前 {MAX_QUERIES} 个查询进行快速测试（总共有 {len(qa_pairs)} 个）")
    qa_pairs = qa_pairs[:MAX_QUERIES]
else:
    logger.info(f"使用全部 {len(qa_pairs)} 个QA对进行测试")

# 准备chunk_id到文档的映射
chunk_id_to_doc = {}
for chunk_idx, chunk in enumerate(chunks):
    text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
    chunk_id = chunk.get('chunk_id') or chunk.get('id') or f'chunk_{chunk_idx}'
    if isinstance(text, str) and len(text.strip()) > 20:
        chunk_id_to_doc[chunk_id] = text.strip()

queries = [qa['question'] for qa in qa_pairs]
logger.info(f"✅ 准备了 {len(queries)} 个查询")

# 准备gold_docs（用于评估）
gold_chunk_ids = [qa.get('chunk_id') for qa in qa_pairs]
gold_texts = [chunk_id_to_doc.get(chunk_id, '') if chunk_id else '' for chunk_id in gold_chunk_ids]

# 初始化HyperAmy
print("\n【步骤2】初始化 HyperAmy...")
try:
    emotion_extractor = Emotion()
    
    # 使用GPU加速（如果可用）
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"✅ 使用GPU加速: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    output_dir = project_root / "outputs" / "three_methods_comparison_monte_cristo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用已有的存储（如果存在）
    storage_path = output_dir / "hyperamy_db"
    if not storage_path.exists():
        logger.error(f"❌ HyperAmy存储不存在: {storage_path}")
        logger.error("   请先运行完整实验或并行索引脚本")
        sys.exit(1)
    
    logger.info(f"✅ 使用已有存储: {storage_path}")
    storage = HyperAmyStorage(persist_path=str(storage_path))
    
    # 读取id_to_content映射
    id_to_content_file = output_dir / "hyperamy_id_to_content.json"
    if not id_to_content_file.exists():
        logger.error(f"❌ id_to_content映射文件不存在: {id_to_content_file}")
        sys.exit(1)
    
    with open(id_to_content_file, 'r', encoding='utf-8') as f:
        id_to_content = json.load(f)
    logger.info(f"✅ 加载了 {len(id_to_content)} 个id->content映射")
    
    # 创建检索器
    from poincare.physics import ParticleProjector
    projector = ParticleProjector()
    hyperamy_retrieval = HyperAmyRetrieval(storage, projector)
    
    print("\n【步骤3】执行HyperAmy检索（10个查询）...")
    
    # 优化：预先提取并缓存查询情绪向量（避免在检索循环中串行等待）
    print("   预先提取查询情绪向量（可缓存复用）...")
    query_emotions_cache_file = output_dir / "query_emotions_cache.json"
    query_emotions_cache = {}
    
    if query_emotions_cache_file.exists():
        # 使用缓存的情绪向量（后续运行）
        logger.info(f"✅ 使用缓存的查询情绪向量: {query_emotions_cache_file}")
        with open(query_emotions_cache_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
            query_emotions_cache = {q: np.array(v) for q, v in cached_data.items()}
    else:
        # 提取并缓存（第一次运行）
        logger.info("   第一次运行，提取查询情绪向量并缓存...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(emotion_extractor.extract, query): query 
                      for query in queries}
            for future in tqdm(as_completed(futures), total=len(queries), desc="提取查询情绪向量"):
                query = futures[future]
                try:
                    emotion_vector = future.result()
                    # 转换为可序列化的格式
                    if isinstance(emotion_vector, np.ndarray):
                        query_emotions_cache[query] = emotion_vector.tolist()
                    else:
                        query_emotions_cache[query] = emotion_vector
                except Exception as e:
                    logger.warning(f"提取查询情绪向量失败 (query={query[:50]}...): {e}")
                    # 使用零向量作为fallback
                    query_emotions_cache[query] = [0.0] * 30
        
        # 保存缓存
        with open(query_emotions_cache_file, 'w', encoding='utf-8') as f:
            json.dump(query_emotions_cache, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 查询情绪向量已缓存到: {query_emotions_cache_file}")
    
    results = []
    hits_at_k = {1: 0, 2: 0, 5: 0, 10: 0}
    
    for i, query in enumerate(tqdm(queries, desc="HyperAmy检索")):
        try:
            # 使用缓存的情绪向量（无需等待API）
            query_emotion = query_emotions_cache.get(query)
            if query_emotion is None:
                # Fallback: 如果缓存中没有，临时提取
                logger.warning(f"缓存中未找到查询情绪向量，临时提取: {query[:50]}...")
                query_emotion = emotion_extractor.extract(query)
            else:
                # 转换为numpy array
                query_emotion = np.array(query_emotion) if not isinstance(query_emotion, np.ndarray) else query_emotion
            
            # 确保emotion_vector是numpy array（ParticleEntity需要numpy array）
            if isinstance(query_emotion, torch.Tensor):
                query_emotion = query_emotion.cpu().numpy()
            elif not isinstance(query_emotion, np.ndarray):
                query_emotion = np.array(query_emotion, dtype=np.float32)
            
            # 计算weight（原始情绪向量的模长）
            weight = float(np.linalg.norm(query_emotion))
            
            # 归一化情绪向量（ParticleEntity期望归一化后的向量）
            if weight > 1e-9:
                normalized_vector = query_emotion / weight
            else:
                normalized_vector = query_emotion.copy()
                weight = 0.0
            
            # 生成查询的entity id
            query_id = f"query_{i}_{hashlib.md5(query.encode('utf-8')).hexdigest()[:16]}"
            
            # 创建ParticleEntity对象用于检索
            query_entity = ParticleEntity(
                entity_id=query_id,
                entity=query[:50],  # 实体名称（使用查询的前50个字符）
                text_id=f"query_{i}",  # 文本ID
                emotion_vector=normalized_vector,  # 归一化后的向量
                weight=weight,  # 原始模长
                speed=0.0,
                temperature=1.0,
                born=0.0
            )
            
            # 检索
            search_results = hyperamy_retrieval.search(query_entity, top_k=5)
            
            # 转换为文档内容，并记录检索到的ID
            retrieved_ids = [r.id for r in search_results]
            docs_retrieved = [id_to_content.get(r.id, '') for r in search_results]
            scores = [r.score for r in search_results]
            
            # 检查是否命中gold_chunk_id
            gold_chunk_id = gold_chunk_ids[i]
            gold_text = gold_texts[i]
            
            # 首先检查gold_chunk_id是否在检索到的ID中（精确匹配）
            hit_at_k = {}
            for k in [1, 2, 5, 10]:
                if k <= len(retrieved_ids):
                    # 方法1: 检查gold_chunk_id是否在检索到的ID中（更准确）
                    found_by_id = gold_chunk_id in retrieved_ids[:k]
                    # 方法2: 检查gold_text是否在前k个结果中（文本匹配，可能不准确）
                    found_by_text = any(gold_text.strip() in doc.strip() or doc.strip() in gold_text.strip() 
                                       for doc in docs_retrieved[:k] if doc)
                    hit_at_k[k] = found_by_id or found_by_text
                    if hit_at_k[k]:
                        hits_at_k[k] += 1
                else:
                    hit_at_k[k] = False
            
            results.append({
                'query': query,
                'gold_chunk_id': gold_chunk_id,
                'gold_text': gold_text[:100] + '...' if len(gold_text) > 100 else gold_text,
                'retrieved_ids': retrieved_ids,  # 添加检索到的ID列表
                'docs_retrieved': [doc[:100] + '...' if len(doc) > 100 else doc for doc in docs_retrieved],
                'scores': scores,
                'hit_at_k': hit_at_k
            })
            
        except Exception as e:
            logger.error(f"HyperAmy检索失败 (query={query[:50]}...): {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'query': query,
                'gold_chunk_id': gold_chunk_ids[i],
                'gold_text': gold_texts[i],
                'docs_retrieved': [],
                'scores': [],
                'hit_at_k': {1: False, 2: False, 5: False, 10: False}
            })
    
    # 打印结果
    print("\n" + "=" * 80)
    print("验证结果统计")
    print("=" * 80)
    print(f"总查询数: {len(queries)}")
    print(f"成功检索: {len([r for r in results if r['docs_retrieved']])}")
    print()
    print("Recall@K 指标:")
    for k in [1, 2, 5, 10]:
        recall = hits_at_k[k] / len(queries) * 100 if queries else 0
        print(f"  Recall@{k}: {hits_at_k[k]}/{len(queries)} ({recall:.1f}%)")
    print()
    
    # 详细结果
    print("详细检索结果:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n查询 {i}: {result['query'][:60]}...")
        print(f"  gold_chunk_id: {result['gold_chunk_id']}")
        print(f"  检索结果数: {len(result['docs_retrieved'])}")
        print(f"  命中情况: Recall@1={result['hit_at_k'][1]}, Recall@2={result['hit_at_k'][2]}, Recall@5={result['hit_at_k'][5]}")
        if result['docs_retrieved']:
            print(f"  第一个结果: {result['docs_retrieved'][0][:80]}...")
    
    # 保存结果
    output_file = output_dir / "hyperamy_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'num_queries': len(queries),
            'recall_at_k': {k: hits_at_k[k] / len(queries) * 100 for k in [1, 2, 5, 10]},
            'hits_at_k': hits_at_k,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 验证结果已保存到: {output_file}")
    
    # 判断验证是否成功
    print("\n" + "=" * 80)
    if hits_at_k[5] > 0:
        print("✅ 验证成功！HyperAmy能够正确检索到gold_chunk_id")
        print(f"   Recall@5 = {hits_at_k[5]}/{len(queries)} ({hits_at_k[5]/len(queries)*100:.1f}%)")
    else:
        print("❌ 验证失败！HyperAmy未能检索到gold_chunk_id")
        print("   请检查检索逻辑和存储数据")
    print("=" * 80)
    
except Exception as e:
    logger.error(f"验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

