#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
四种检索方法对比实验 - 使用Monte Cristo数据集

方法：
1. hyperamy: 纯情绪检索（使用poincare双曲空间，LLM抽取情绪向量）
2. hyperamy_emos: 纯情绪检索（使用poincare双曲空间，emos模型抽取情绪向量）
3. hipporag: 纯语义检索（标准HippoRAG）
4. fusion: 语义+情绪混合检索（HippoRAGEnhanced，最佳配置: harmonic_none_0.4）

数据集：
- 训练数据: data/training/monte_cristo_train_full.jsonl
- QA数据: data/public_benchmark/monte_cristo_qa_full.json (50个QA对)
"""
import sys
import os
import json
import numpy as np
import logging
import hashlib
import uuid
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.config import API_KEY, BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
# from sentiment.hipporag_enhanced import HippoRAGEnhanced  # 暂不使用Fusion
# from sentiment.fusion_strategies import FusionStrategy, NormalizationStrategy  # 暂不使用Fusion
from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from poincare.storage import HyperAmyStorage
from poincare.retrieval import HyperAmyRetrieval
from particle.particle import ParticleEntity
from particle.emotion import Emotion, EMOTIONS
from particle.emotion_v3 import EmotionV3
from sentence_transformers import SentenceTransformer

# 设置环境变量
os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['API_KEY'] = API_KEY

# 配置日志
log_file = project_root / 'test_vibe_search_experiment.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型配置
llm_model_name = DEFAULT_MODEL
llm_base_url = BASE_URL
embedding_model_name = f"VLLM/{DEFAULT_EMBEDDING_MODEL}"
embedding_base_url = API_URL_EMBEDDINGS

print("=" * 80)
print("Vibe Search / Emotional Retrieval 实验 - Monte Cristo数据集")
print("=" * 80)
print("方法说明：")
print("  1. HippoRAG: 纯语义检索（标准HippoRAG）")
print("  2. HyperAmy (LLM): 纯情绪检索（使用poincare双曲空间，LLM抽取情绪向量）")
print("  3. Hybrid (Dynamic v2): 动态权重混合检索（Weighted RRF + Adaptive Weighting）")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 加载数据集
print("\n【步骤1】加载数据集...")
chunks_file = project_root / "data" / "training" / "monte_cristo_train_full.jsonl"
vibe_file = project_root / "data" / "public_benchmark" / "monte_cristo_vibe_search.json"

if not chunks_file.exists():
    print(f"❌ 训练数据文件不存在: {chunks_file}")
    sys.exit(1)

if not vibe_file.exists():
    print(f"❌ Vibe Search数据集文件不存在: {vibe_file}")
    print(f"   数据集可能还在生成中，请运行: python scripts/generate_vibe_dataset.py")
    sys.exit(1)

# 加载chunks
chunks = []
with open(chunks_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))
print(f"✅ 加载了 {len(chunks)} 个chunks")

# 加载Vibe Search数据
with open(vibe_file, 'r', encoding='utf-8') as f:
    vibe_data = json.load(f)
vibe_queries = vibe_data.get('data', [])
print(f"✅ 加载了 {len(vibe_queries)} 个Vibe查询")
print(f"   数据集名称: {vibe_data.get('dataset_name', 'N/A')}")
print(f"   情绪密度阈值: {vibe_data.get('emotion_intensity_threshold', 'N/A')}/10")

# 统计情绪类型分布
from collections import Counter
emotion_tags = Counter([q.get('emotion_tag', 'Unknown') for q in vibe_queries])
print(f"   情绪类型分布:")
for emotion, count in emotion_tags.items():
    print(f"     - {emotion}: {count}")

# 转换为qa_pairs格式（兼容现有代码）
qa_pairs = []
for vq in vibe_queries:
    qa_pairs.append({
        'question': vq.get('query', ''),
        'chunk_id': vq.get('gold_chunk_id', ''),
        'answer': '',  # Vibe数据集没有answer字段
        'gold_text': vq.get('gold_text', ''),
        'emotion_tag': vq.get('emotion_tag', ''),
        'emotion_intensity': vq.get('emotion_intensity', 0.0),
        'requires_emotional_sensitivity': True  # Vibe查询都是情绪敏感的
    })
print(f"✅ 转换了 {len(qa_pairs)} 个QA对（用于兼容现有代码）")

# 限制测试数量（可选，0表示使用全部）
MAX_QUERIES = 0  # 0=全部50个，也可以改为10进行快速测试
if MAX_QUERIES > 0 and len(qa_pairs) > MAX_QUERIES:
    print(f"\n⚠️  限制为前 {MAX_QUERIES} 个查询进行测试（总共有 {len(qa_pairs)} 个）")
    qa_pairs = qa_pairs[:MAX_QUERIES]
else:
    print(f"使用全部 {len(qa_pairs)} 个QA对进行测试")

# 准备文档列表（用于索引）
docs = []
chunk_id_to_doc = {}
for chunk_idx, chunk in enumerate(chunks):
    # 尝试多个可能的字段名（按优先级：input > text > content > chunk_text）
    text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
    # 训练数据中没有chunk_id字段，需要根据索引生成（格式：chunk_{index}）
    chunk_id = chunk.get('chunk_id') or chunk.get('id') or f'chunk_{chunk_idx}'
    
    # 确保text是字符串且有一定长度
    if isinstance(text, str) and len(text.strip()) > 20:
        docs.append(text.strip())
        # 始终保存chunk_id到文档的映射（用于后续匹配QA数据）
        chunk_id_to_doc[chunk_id] = text.strip()
    else:
        logger.warning(f"跳过无效chunk (索引={chunk_idx}, chunk_id={chunk_id}): text长度={len(text) if isinstance(text, str) else 'N/A'}")

print(f"✅ 准备了 {len(docs)} 个文档用于索引")
if len(docs) == 0:
    print("⚠️  警告：没有加载到任何文档！")
    print(f"   检查：chunks数量={len(chunks)}")
    if chunks:
        print(f"   第一个chunk的字段: {list(chunks[0].keys())}")
        first_text = chunks[0].get('text') or chunks[0].get('input') or chunks[0].get('content', '')
        print(f"   第一个chunk的text字段值: {type(first_text)}, 长度={len(first_text) if isinstance(first_text, str) else 'N/A'}")

queries = [qa['question'] for qa in qa_pairs]
print(f"✅ 准备了 {len(queries)} 个查询")

# 准备gold_docs（用于评估）
gold_docs = []
for qa in qa_pairs:
    chunk_id = qa.get('chunk_id')
    gold_text = chunk_id_to_doc.get(chunk_id) if chunk_id else None
    if gold_text:
        gold_docs.append([gold_text])
    else:
        gold_docs.append([])

# 创建输出目录
output_dir = project_root / "outputs" / "vibe_search_experiment"
output_dir.mkdir(parents=True, exist_ok=True)

results = []

# ========== 方法1: HippoRAG (纯语义) ==========
print("\n【步骤2】初始化 HippoRAG (纯语义检索)...")
save_dir_hipporag = output_dir / "hipporag"
save_dir_hipporag.mkdir(exist_ok=True)

config_hipporag = BaseConfig(
    save_dir=str(save_dir_hipporag),
    llm_base_url=llm_base_url,
    llm_name=llm_model_name,
    embedding_model_name=embedding_model_name,
    embedding_base_url=embedding_base_url,
    force_index_from_scratch=False,  # 如果索引已存在，复用索引以加速
    retrieval_top_k=5,
)

try:
    hipporag = HippoRAG(
        global_config=config_hipporag,
        llm_model_name=llm_model_name,
        llm_base_url=llm_base_url,
        embedding_model_name=embedding_model_name,
        embedding_base_url=embedding_base_url
        # 注意：服务器上的HippoRAG版本不支持openie_max_workers参数
    )
    print("✅ HippoRAG 初始化成功")
    
    print("\n【步骤3】索引文档（HippoRAG）...")
    hipporag.index(docs=docs)
    print("✅ HippoRAG 索引完成")
    
    print("\n【步骤4】HippoRAG 检索...")
    hipporag_results, hipporag_eval = hipporag.retrieve(
        queries=queries,
        num_to_retrieve=5,
        gold_docs=gold_docs
    )
    print("✅ HippoRAG 检索完成")
    if hipporag_eval:
        print(f"   检索评估指标: {hipporag_eval}")
    hipporag_available = True
except Exception as e:
    logger.error(f"HippoRAG 失败: {e}")
    import traceback
    traceback.print_exc()
    hipporag_available = False
    hipporag_results = None
    hipporag_eval = None

# # ========== 方法2: Fusion (语义+情绪混合) ==========
# print("\n【步骤5】初始化 HippoRAGEnhanced (Fusion: 语义+情绪混合)...")
# save_dir_fusion = output_dir / "fusion"
# save_dir_fusion.mkdir(exist_ok=True)
# 
# config_fusion = BaseConfig(
#     save_dir=str(save_dir_fusion),
#     llm_base_url=llm_base_url,
#     llm_name=llm_model_name,
#     embedding_model_name=embedding_model_name,
#     embedding_base_url=embedding_base_url,
#     force_index_from_scratch=False,  # 如果索引已存在，复用索引以加速
#     retrieval_top_k=5,
# )
# 
# try:
#     fusion = HippoRAGEnhanced(
#         global_config=config_fusion,
#         enable_sentiment=True,
#         sentiment_weight=0.4,  # 最佳权重（基于网格搜索结果）
#         sentiment_model_name=llm_model_name,
#         max_workers=10,  # 使用并发优化
#         fusion_strategy=FusionStrategy.HARMONIC,  # 最佳策略
#         normalization_strategy=NormalizationStrategy.NONE  # 最佳归一化
#     )
#     print("✅ Fusion 初始化成功（最佳配置: harmonic_none_0.4, max_workers=10）")
#     
#     print("\n【步骤6】索引文档（Fusion）- 使用并发优化...")
#     fusion.index(docs=docs)
#     print("✅ Fusion 索引完成")
#     
#     print("\n【步骤7】Fusion 检索...")
#     fusion_results, fusion_eval = fusion.retrieve(
#         queries=queries,
#         num_to_retrieve=5,
#         gold_docs=gold_docs
#     )
#     print("✅ Fusion 检索完成")
#     if fusion_eval:
#         print(f"   检索评估指标: {fusion_eval}")
#     fusion_available = True
# except Exception as e:
#     logger.error(f"Fusion 失败: {e}")
#     import traceback
#     traceback.print_exc()
#     fusion_available = False
#     fusion_results = None
#     fusion_eval = None

# Fusion部分被注释掉了，初始化变量避免NameError
fusion_available = False
fusion_results = None
fusion_eval = None

# ========== 方法3: HyperAmy (纯情绪) ==========
print("\n【步骤8】初始化 HyperAmy (纯情绪检索)...")
try:
    # 使用poincare的HyperAmyRetrieval
    emotion_extractor = Emotion()
    
    # 使用GPU加速SentenceTransformer（如果可用）
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"✅ 使用GPU加速: {torch.cuda.get_device_name(0)}")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # 创建存储
    storage_path = output_dir / "hyperamy_db"
    id_to_content_file = output_dir / "hyperamy_id_to_content.json"
    
    # 维护id->content映射（因为Point没有content字段，SearchResult也没有）
    # 使用与chunk_id_to_doc相同的chunk_id格式，确保可以正确映射
    id_to_content = {}
    
    # 检查是否已有可用的索引（主存储路径或并行索引结果）
    use_existing_index = False
    
    # 1. 首先检查主存储路径是否已完成
    if storage_path.exists() and id_to_content_file.exists():
        try:
            # 尝试读取映射文件
            with open(id_to_content_file, 'r', encoding='utf-8') as f:
                id_to_content = json.load(f)
            
            # 检查主存储的点数（需要等待ChromaDB完成写入，可能需要多次尝试）
            import time
            storage_check_retries = 3
            storage_count = 0
            for retry in range(storage_check_retries):
                try:
                    temp_storage = HyperAmyStorage(persist_path=str(storage_path))
                    storage_count = temp_storage.collection.count()
                    if storage_count > 0:
                        break
                except Exception as e:
                    if retry < storage_check_retries - 1:
                        logger.info(f"   等待ChromaDB就绪（重试 {retry+1}/{storage_check_retries}）...")
                        time.sleep(5)
                    else:
                        logger.warning(f"   无法读取存储: {e}")
            
            if storage_count >= len(id_to_content) and storage_count >= 9000:  # 至少9000个点才认为可用
                logger.info(f"✅ 检测到HyperAmy索引已完成（{storage_count}个点），直接使用现有索引")
                storage = HyperAmyStorage(persist_path=str(storage_path))
                use_existing_index = True
                logger.info(f"✅ 跳过情绪向量提取，直接使用现有索引进行检索")
            else:
                logger.info(f"   现有索引点数不足（{storage_count}/{len(id_to_content)}），将重新索引")
        except Exception as e:
            logger.warning(f"⚠️  读取现有索引失败: {e}，将重新索引")
    
    if not use_existing_index:
        # 如果没有使用并行索引，则正常初始化存储并提取情绪向量
        storage = HyperAmyStorage(persist_path=str(storage_path))
        
        # 提取情绪向量并存储点（并发优化）
        print("   提取情绪向量并存储点（并发处理）...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def extract_emotion_for_chunk(chunk_data):
            """提取单个chunk的情绪向量（用于并发调用）"""
            chunk_idx, chunk = chunk_data
            # 使用与docs加载相同的字段检查逻辑（input > text > content > chunk_text）
            text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
            # 生成chunk_id（与docs加载逻辑完全一致，确保可以匹配QA数据中的chunk_id）
            chunk_id = chunk.get('chunk_id') or chunk.get('id') or f'chunk_{chunk_idx}'
            
            # 确保text是字符串且有一定长度
            if isinstance(text, str) and len(text.strip()) > 20:
                try:
                    emotion_vector = emotion_extractor.extract(text)
                    # 确保emotion_vector是numpy array（ParticleEntity需要numpy array）
                    import torch
                    if isinstance(emotion_vector, torch.Tensor):
                        emotion_vector = emotion_vector.cpu().numpy()
                    elif not isinstance(emotion_vector, np.ndarray):
                        emotion_vector = np.array(emotion_vector, dtype=np.float32)
                    
                    # 计算weight（原始情绪向量的模长）
                    weight = float(np.linalg.norm(emotion_vector))
                    
                    # 归一化情绪向量（ParticleEntity期望归一化后的向量）
                    if weight > 1e-9:
                        normalized_vector = emotion_vector / weight
                    else:
                        normalized_vector = emotion_vector.copy()
                        weight = 0.0
                    
                    # 使用chunk_id作为entity_id（确保与chunk_id_to_doc中的key一致）
                    entity_id = chunk_id  # chunk_id已经是格式化的（chunk_0, chunk_1, ...或chunk_3806等）
                    
                    return {
                        'success': True,
                        'chunk_idx': chunk_idx,
                        'chunk_id': chunk_id,
                        'entity_id': entity_id,
                        'emotion_vector': normalized_vector,  # 归一化后的向量
                        'weight': weight,  # 原始模长
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
        
        # 使用ThreadPoolExecutor并发处理
        max_workers = 10  # 并发线程数
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
                              desc="提取情绪向量（并发）"):
                try:
                    result = future.result()
                    results_list.append(result)
                except Exception as e:
                    chunk_idx = future_to_chunk[future]
                    logger.error(f"并发处理异常 (索引={chunk_idx}): {e}")
                    skipped_count += 1
        
            # 按照原始顺序整理结果并存储
            results_list.sort(key=lambda x: x['chunk_idx'])
            
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
                        purity=0.5,  # 归一化纯度
                        tau_v=86400.0,  # 速度衰减时间常数
                        tau_T=86400.0,  # 温度冷却时间常数
                        born=time.time()  # 创建时间（使用当前时间戳）
                    )
                    storage.upsert_entity(entity)
                    stored_points += 1
                else:
                    skipped_count += 1
                    if result['chunk_idx'] < 5:  # 只记录前5个跳过的chunks作为示例
                        logger.debug(f"跳过chunk (索引={result['chunk_idx']}): {result['error']}")
            
            # 保存id_to_content映射到文件（供后续使用）
            id_to_content_file = output_dir / "hyperamy_id_to_content.json"
            with open(id_to_content_file, 'w', encoding='utf-8') as f:
                json.dump(id_to_content, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 保存了id->content映射到: {id_to_content_file}")
            
            print(f"✅ HyperAmy 存储初始化完成（存储了 {stored_points} 个点，跳过了 {skipped_count} 个无效chunks）")
    else:
        # 使用并行索引时，已经加载了id_to_content，不需要重新提取
        storage = HyperAmyStorage(persist_path=str(storage_path))
        stored_points = len(id_to_content)
        logger.info(f"✅ 使用并行索引结果（{stored_points}个点）")
    
    # 创建检索器
    from poincare.projector import ParticleProjector
    projector = ParticleProjector()
    hyperamy_retrieval = HyperAmyRetrieval(storage, projector)
    
    print("\n【步骤9】HyperAmy-Hybrid 混合检索（语义+情绪重排序）...")
    hyperamy_results = []
    for i, query in enumerate(tqdm(queries, desc="HyperAmy-Hybrid检索")):
        try:
            # ========== 第一阶段：获取Top-100语义候选（从HippoRAG） ==========
            if hipporag_available:
                # 使用HippoRAG获取Top-100语义相关的结果
                semantic_results_100 = hipporag.retrieve(
                    queries=[query],
                    num_to_retrieve=100  # 获取100个语义候选
                )
                if semantic_results_100 and len(semantic_results_100) > 0:
                    semantic_docs = semantic_results_100[0].docs  # 文档列表
                    semantic_scores = semantic_results_100[0].doc_scores  # 分数数组
                else:
                    logger.warning(f"HyperAmy-Hybrid: 无法获取语义候选（query={query[:50]}），跳过")
                    from hipporag.utils.misc_utils import QuerySolution
                    hyperamy_results.append(QuerySolution(
                        question=query,
                        docs=[],
                        doc_scores=np.array([])
                    ))
                    continue
            else:
                logger.warning(f"HyperAmy-Hybrid: HippoRAG不可用，跳过混合检索")
                from hipporag.utils.misc_utils import QuerySolution
                hyperamy_results.append(QuerySolution(
                    question=query,
                    docs=[],
                    doc_scores=np.array([])
                ))
                continue
            
            # ========== 第二阶段：准备查询情绪向量 ==========
            # 提取查询的情绪向量
            query_emotion = emotion_extractor.extract(query)
            # 确保emotion_vector是numpy array（ParticleEntity需要numpy array）
            import torch
            if isinstance(query_emotion, torch.Tensor):
                query_emotion_np = query_emotion.cpu().numpy().astype(np.float32)
            elif isinstance(query_emotion, np.ndarray):
                query_emotion_np = query_emotion.astype(np.float32)
            else:
                query_emotion_np = np.array(query_emotion, dtype=np.float32)
            
            # 维度验证：确保查询向量维度与存储的向量维度一致（应该是28维，对应28种情绪）
            expected_dim = len(EMOTIONS)
            if len(query_emotion_np) != expected_dim:
                logger.warning(
                    f"查询向量维度异常：{len(query_emotion_np)}，预期：{expected_dim}。"
                    f"尝试自动修复（截断或填充）."
                )
                if len(query_emotion_np) > expected_dim:
                    query_emotion_np = query_emotion_np[:expected_dim]
                else:
                    padding = np.zeros(expected_dim - len(query_emotion_np), dtype=np.float32)
                    query_emotion_np = np.concatenate((query_emotion_np, padding))
            
            query_emotion = query_emotion_np
            
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
                purity=0.5,  # 归一化纯度
                tau_v=86400.0,  # 速度衰减时间常数
                tau_T=86400.0,  # 温度冷却时间常数
                born=time.time()  # 创建时间（使用当前时间戳）
            )
            
            # ========== 第三阶段：混合检索（Hyper-Rerank） ==========
            search_results = hyperamy_retrieval.search_hybrid(
                query_text=query,
                query_entity=query_entity,
                semantic_docs=semantic_docs,
                semantic_scores=semantic_scores,
                id_to_content=id_to_content,
                top_k=5,
                alpha=0.8  # 语义分数权重0.8，HyperAmy分数权重0.2
            )
            # 转换为QuerySolution格式
            # SearchResult没有content字段，需要从id_to_content映射中获取
            from hipporag.utils.misc_utils import QuerySolution
            docs_retrieved = [id_to_content.get(r.id, '') for r in search_results]
            scores = [r.score for r in search_results]
            hyperamy_results.append(QuerySolution(
                question=query,
                docs=docs_retrieved,
                doc_scores=np.array(scores)
            ))
        except Exception as e:
            logger.error(f"HyperAmy检索失败 (query={query[:50]}...): {e}")
            # 创建空结果
            from hipporag.utils.misc_utils import QuerySolution
            hyperamy_results.append(QuerySolution(
                question=query,
                docs=[],
                doc_scores=np.array([])
            ))
    
    print("✅ HyperAmy 检索完成")
    hyperamy_available = True
except Exception as e:
    logger.error(f"HyperAmy 失败: {e}")
    import traceback
    traceback.print_exc()
    hyperamy_available = False
    hyperamy_results = None

# ========== 方法4: HyperAmy-Emos (使用emos模型) ==========
print("\n【步骤10】初始化 HyperAmy-Emos (纯情绪检索，使用emos模型)...")
try:
    # 使用emos模型的EmotionV3
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"✅ 使用GPU加速: {torch.cuda.get_device_name(0)}")
    
    emotion_extractor_emos = EmotionV3(
        emos_checkpoint_path=None,  # 使用默认路径
        device=device
    )
    
    # 创建存储
    storage_path_emos = output_dir / "hyperamy_emos_db"
    id_to_content_file_emos = output_dir / "hyperamy_emos_id_to_content.json"
    
    # 维护id->content映射
    id_to_content_emos = {}
    
    # 检查是否已有索引
    use_existing_index_emos = False
    if storage_path_emos.exists() and id_to_content_file_emos.exists():
        try:
            with open(id_to_content_file_emos, 'r', encoding='utf-8') as f:
                id_to_content_emos = json.load(f)
            temp_storage = HyperAmyStorage(persist_path=str(storage_path_emos))
            storage_count = temp_storage.collection.count()
            if storage_count > 0:
                use_existing_index_emos = True
                logger.info(f"✅ 找到已存在的Emos索引: {storage_count}个点")
        except Exception as e:
            logger.warning(f"无法读取已存在的Emos索引: {e}")
    
    if not use_existing_index_emos:
        # 创建新存储
        storage_emos = HyperAmyStorage(persist_path=str(storage_path_emos))
        
        print("【步骤11】提取情绪向量（Emos模型）- 使用并发处理...")
        
        def extract_emotion_for_chunk_emos(chunk_info):
            chunk_idx, chunk = chunk_info
            try:
                chunk_id = chunk.get('chunk_id') or chunk.get('id') or f'chunk_{chunk_idx}'
                text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
                
                if not isinstance(text, str) or len(text.strip()) < 20:
                    return {
                        'success': False,
                        'chunk_idx': chunk_idx,
                        'chunk_id': chunk_id,
                        'error': f"text长度不足或无效"
                    }
                
                # 使用EmotionV3提取实体情绪向量
                nodes = emotion_extractor_emos.process(
                    text=text,
                    text_id=chunk_id
                )
                
                if not nodes:
                    return {
                        'success': False,
                        'chunk_idx': chunk_idx,
                        'chunk_id': chunk_id,
                        'error': "未提取到实体情绪向量"
                    }
                
                # 取第一个实体（或者可以合并所有实体）
                # 这里为了简化，我们使用第一个实体作为chunk的代表
                main_node = nodes[0]
                
                # 计算weight（向量的L2范数）
                weight = float(np.linalg.norm(main_node.emotion_vector))
                
                # 归一化向量
                if weight > 1e-9:
                    normalized_vector = main_node.emotion_vector / weight
                else:
                    normalized_vector = main_node.emotion_vector.copy()
                    weight = 0.0
                
                return {
                    'success': True,
                    'chunk_idx': chunk_idx,
                    'chunk_id': chunk_id,
                    'entity_id': main_node.entity_id,
                    'emotion_vector': normalized_vector,
                    'weight': weight,
                    'text': text
                }
            except Exception as e:
                return {
                    'success': False,
                    'chunk_idx': chunk_idx,
                    'chunk_id': chunk_id,
                    'error': str(e)
                }
        
        # 并发处理
        max_workers = 10
        stored_points_emos = 0
        skipped_count_emos = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(extract_emotion_for_chunk_emos, (chunk_idx, chunk)): chunk_idx
                for chunk_idx, chunk in enumerate(chunks)
            }
            
            results_list_emos = []
            for future in tqdm(as_completed(future_to_chunk), 
                              total=len(chunks), 
                              desc="提取情绪向量（Emos模型，并发）"):
                try:
                    result = future.result()
                    results_list_emos.append(result)
                except Exception as e:
                    chunk_idx = future_to_chunk[future]
                    logger.error(f"并发处理异常 (索引={chunk_idx}): {e}")
                    skipped_count_emos += 1
            
            # 整理结果并存储
            results_list_emos.sort(key=lambda x: x['chunk_idx'])
            
            for result in results_list_emos:
                if result['success']:
                    id_to_content_emos[result['entity_id']] = result['text']
                    
                    entity = ParticleEntity(
                        entity_id=result['entity_id'],
                        entity=f"chunk_{result['chunk_idx']}",
                        text_id=result['chunk_id'],
                        emotion_vector=result['emotion_vector'],
                        weight=result['weight'],
                        speed=0.0,
                        temperature=1.0,
                        purity=0.5,  # 归一化纯度
                        tau_v=86400.0,  # 速度衰减时间常数
                        tau_T=86400.0,  # 温度冷却时间常数
                        born=time.time()  # 创建时间（使用当前时间戳）
                    )
                    storage_emos.upsert_entity(entity)
                    stored_points_emos += 1
                else:
                    skipped_count_emos += 1
            
            # 保存id_to_content映射
            with open(id_to_content_file_emos, 'w', encoding='utf-8') as f:
                json.dump(id_to_content_emos, f, ensure_ascii=False, indent=2)
            
            print(f"✅ HyperAmy-Emos 存储初始化完成（存储了 {stored_points_emos} 个点，跳过了 {skipped_count_emos} 个无效chunks）")
    else:
        storage_emos = HyperAmyStorage(persist_path=str(storage_path_emos))
        stored_points_emos = len(id_to_content_emos)
        logger.info(f"✅ 使用已存在的Emos索引（{stored_points_emos}个点）")
    
    # 创建检索器
    from poincare.projector import ParticleProjector
    projector_emos = ParticleProjector()
    hyperamy_emos_retrieval = HyperAmyRetrieval(storage_emos, projector_emos)
    
    print("\n【步骤12】HyperAmy-Emos 检索...")
    hyperamy_emos_results = []
    for i, query in enumerate(tqdm(queries, desc="HyperAmy-Emos检索")):
        try:
            # 使用EmotionV3提取查询的情绪向量
            query_nodes = emotion_extractor_emos.process(
                text=query,
                text_id=f"query_{i}"
            )
            
            if not query_nodes:
                # 创建空结果
                from hipporag.utils.misc_utils import QuerySolution
                hyperamy_emos_results.append(QuerySolution(
                    question=query,
                    docs=[],
                    doc_scores=np.array([])
                ))
                continue
            
            # 使用第一个实体作为查询代表
            query_node = query_nodes[0]
            query_emotion = query_node.emotion_vector
            
            # 确保emotion_vector是numpy array
            import torch
            if isinstance(query_emotion, torch.Tensor):
                query_emotion_np = query_emotion.cpu().numpy().astype(np.float32)
            elif isinstance(query_emotion, np.ndarray):
                query_emotion_np = query_emotion.astype(np.float32)
            else:
                query_emotion_np = np.array(query_emotion, dtype=np.float32)
            
            # 维度验证：确保查询向量维度与存储的向量维度一致（应该是28维，对应28种情绪）
            expected_dim = len(EMOTIONS)
            if len(query_emotion_np) != expected_dim:
                logger.warning(
                    f"查询向量维度异常：{len(query_emotion_np)}，预期：{expected_dim}。"
                    f"尝试自动修复（截断或填充）."
                )
                if len(query_emotion_np) > expected_dim:
                    query_emotion_np = query_emotion_np[:expected_dim]
                else:
                    padding = np.zeros(expected_dim - len(query_emotion_np), dtype=np.float32)
                    query_emotion_np = np.concatenate((query_emotion_np, padding))
            
            # 计算weight和归一化（使用修复后的向量）
            weight = float(np.linalg.norm(query_emotion_np))
            if weight > 1e-9:
                normalized_vector = query_emotion_np / weight
            else:
                normalized_vector = query_emotion_np.copy()
                weight = 0.0
            
            # 创建ParticleEntity对象用于检索
            query_entity = ParticleEntity(
                entity_id=query_node.entity_id,
                entity=query[:50],
                text_id=f"query_{i}",
                emotion_vector=normalized_vector,
                weight=weight,
                speed=0.0,
                temperature=1.0,
                purity=0.5,  # 归一化纯度
                tau_v=86400.0,  # 速度衰减时间常数
                tau_T=86400.0,  # 温度冷却时间常数
                born=time.time()  # 创建时间（使用当前时间戳）
            )
            
            # 检索
            search_results = hyperamy_emos_retrieval.search(query_entity, top_k=5)
            
            # 转换为QuerySolution格式
            from hipporag.utils.misc_utils import QuerySolution
            docs_retrieved = [id_to_content_emos.get(r.id, '') for r in search_results]
            scores = [r.score for r in search_results]
            hyperamy_emos_results.append(QuerySolution(
                question=query,
                docs=docs_retrieved,
                doc_scores=np.array(scores)
            ))
        except Exception as e:
            logger.error(f"HyperAmy-Emos检索失败 (query={query[:50]}...): {e}")
            from hipporag.utils.misc_utils import QuerySolution
            hyperamy_emos_results.append(QuerySolution(
                question=query,
                docs=[],
                doc_scores=np.array([])
            ))
    
    print("✅ HyperAmy-Emos 检索完成")
    hyperamy_emos_available = True
except Exception as e:
    logger.error(f"HyperAmy-Emos 失败: {e}")
    import traceback
    traceback.print_exc()
    hyperamy_emos_available = False
    hyperamy_emos_results = None

# ========== 结果对比和保存 ==========
print("\n【步骤10】结果对比和保存...")
comparison_results = []

for i, qa_pair in enumerate(qa_pairs):
    gold_chunk_id = qa_pair.get('chunk_id', '')
    gold_text = chunk_id_to_doc.get(gold_chunk_id) if gold_chunk_id else None
    
    result = {
        'question': qa_pair.get('question', ''),
        'gold_answer': qa_pair.get('answer', ''),
        'gold_chunk_id': gold_chunk_id,
        'gold_text': gold_text[:200] + '...' if gold_text and len(gold_text) > 200 else gold_text,
        'requires_emotional_sensitivity': qa_pair.get('requires_emotional_sensitivity', False),
    }
    
    # HippoRAG结果
    if hipporag_available and hipporag_results and i < len(hipporag_results):
        hipporag_result = hipporag_results[i]
        result['hipporag'] = {
            'docs': hipporag_result.docs if hasattr(hipporag_result, 'docs') else [],
            'scores': hipporag_result.doc_scores.tolist() if hasattr(hipporag_result, 'doc_scores') else [],
            'available': True,
            'hit': gold_text in (hipporag_result.docs if hasattr(hipporag_result, 'docs') else [])
        }
    else:
        result['hipporag'] = {'available': False, 'hit': False}
    
    # Fusion结果
    if fusion_available and fusion_results and i < len(fusion_results):
        fusion_result = fusion_results[i]
        result['fusion'] = {
            'docs': fusion_result.docs if hasattr(fusion_result, 'docs') else [],
            'scores': fusion_result.doc_scores.tolist() if hasattr(fusion_result, 'doc_scores') else [],
            'available': True,
            'hit': gold_text in (fusion_result.docs if hasattr(fusion_result, 'docs') else [])
        }
    else:
        result['fusion'] = {'available': False, 'hit': False}
    
    # HyperAmy结果
    if hyperamy_available and hyperamy_results and i < len(hyperamy_results):
        hyperamy_result = hyperamy_results[i]
        result['hyperamy'] = {
            'docs': hyperamy_result.docs if hasattr(hyperamy_result, 'docs') else [],
            'scores': hyperamy_result.doc_scores.tolist() if hasattr(hyperamy_result, 'doc_scores') else [],
            'available': True,
            'hit': gold_text in (hyperamy_result.docs if hasattr(hyperamy_result, 'docs') else [])
        }
    else:
        result['hyperamy'] = {'available': False, 'hit': False}
    
    # HyperAmy-Emos结果
    if hyperamy_emos_available and hyperamy_emos_results and i < len(hyperamy_emos_results):
        hyperamy_emos_result = hyperamy_emos_results[i]
        result['hyperamy_emos'] = {
            'docs': hyperamy_emos_result.docs if hasattr(hyperamy_emos_result, 'docs') else [],
            'scores': hyperamy_emos_result.doc_scores.tolist() if hasattr(hyperamy_emos_result, 'doc_scores') else [],
            'available': True,
            'hit': gold_text in (hyperamy_emos_result.docs if hasattr(hyperamy_emos_result, 'docs') else [])
        }
    else:
        result['hyperamy_emos'] = {'available': False, 'hit': False}
    
    comparison_results.append(result)

# 保存结果
result_file = output_dir / "comparison_results.json"
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, ensure_ascii=False, indent=2)
print(f"✅ 结果已保存到: {result_file}")

# 统计信息
print("\n" + "=" * 80)
print("检索命中率对比")
print("=" * 80)

total = len(comparison_results)

hipporag_hits = sum(1 for r in comparison_results if r.get('hipporag', {}).get('hit', False))
fusion_hits = sum(1 for r in comparison_results if r.get('fusion', {}).get('hit', False))
hyperamy_hits = sum(1 for r in comparison_results if r.get('hyperamy', {}).get('hit', False))
hyperamy_emos_hits = sum(1 for r in comparison_results if r.get('hyperamy_emos', {}).get('hit', False))

print(f"HippoRAG (纯语义):     {hipporag_hits}/{total} ({100*hipporag_hits/total:.1f}%)")
print(f"Fusion (混合):         {fusion_hits}/{total} ({100*fusion_hits/total:.1f}%)")
print(f"HyperAmy (纯情绪-LLM): {hyperamy_hits}/{total} ({100*hyperamy_hits/total:.1f}%)")
print(f"HyperAmy-Emos (纯情绪): {hyperamy_emos_hits}/{total} ({100*hyperamy_emos_hits/total:.1f}%)")
print("=" * 80)

# 按情绪敏感性分组统计
emotional_qs = [r for r in comparison_results if r.get('requires_emotional_sensitivity', False)]
non_emotional_qs = [r for r in comparison_results if not r.get('requires_emotional_sensitivity', False)]

if emotional_qs:
    print("\n【情绪敏感性查询统计】")
    print("-" * 80)
    emotional_total = len(emotional_qs)
    emotional_hipporag_hits = sum(1 for r in emotional_qs if r.get('hipporag', {}).get('hit', False))
    emotional_fusion_hits = sum(1 for r in emotional_qs if r.get('fusion', {}).get('hit', False))
    emotional_hyperamy_hits = sum(1 for r in emotional_qs if r.get('hyperamy', {}).get('hit', False))
    emotional_hyperamy_emos_hits = sum(1 for r in emotional_qs if r.get('hyperamy_emos', {}).get('hit', False))
    
    print(f"查询数: {emotional_total}")
    print(f"HippoRAG:       {emotional_hipporag_hits}/{emotional_total} ({100*emotional_hipporag_hits/emotional_total:.1f}%)")
    print(f"Fusion:         {emotional_fusion_hits}/{emotional_total} ({100*emotional_fusion_hits/emotional_total:.1f}%)")
    print(f"HyperAmy (LLM): {emotional_hyperamy_hits}/{emotional_total} ({100*emotional_hyperamy_hits/emotional_total:.1f}%)")
    print(f"HyperAmy-Emos:  {emotional_hyperamy_emos_hits}/{emotional_total} ({100*emotional_hyperamy_emos_hits/emotional_total:.1f}%)")

if non_emotional_qs:
    print("\n【非情绪敏感性查询统计】")
    print("-" * 80)
    non_emotional_total = len(non_emotional_qs)
    non_emotional_hipporag_hits = sum(1 for r in non_emotional_qs if r.get('hipporag', {}).get('hit', False))
    non_emotional_fusion_hits = sum(1 for r in non_emotional_qs if r.get('fusion', {}).get('hit', False))
    non_emotional_hyperamy_hits = sum(1 for r in non_emotional_qs if r.get('hyperamy', {}).get('hit', False))
    non_emotional_hyperamy_emos_hits = sum(1 for r in non_emotional_qs if r.get('hyperamy_emos', {}).get('hit', False))
    
    print(f"查询数: {non_emotional_total}")
    print(f"HippoRAG:       {non_emotional_hipporag_hits}/{non_emotional_total} ({100*non_emotional_hipporag_hits/non_emotional_total:.1f}%)")
    print(f"Fusion:         {non_emotional_fusion_hits}/{non_emotional_total} ({100*non_emotional_fusion_hits/non_emotional_total:.1f}%)")
    print(f"HyperAmy (LLM): {non_emotional_hyperamy_hits}/{non_emotional_total} ({100*non_emotional_hyperamy_hits/non_emotional_total:.1f}%)")
    print(f"HyperAmy-Emos:  {non_emotional_hyperamy_emos_hits}/{non_emotional_total} ({100*non_emotional_hyperamy_emos_hits/non_emotional_total:.1f}%)")

# 打印评估指标
if hipporag_eval:
    print("\nHippoRAG 评估指标:")
    for key, value in hipporag_eval.items():
        print(f"  {key}: {value}")

if fusion_eval:
    print("\nFusion 评估指标:")
    for key, value in fusion_eval.items():
        print(f"  {key}: {value}")

print(f"\n✅ 实验完成！")
print(f"结果文件: {result_file}")
print(f"日志文件: {log_file}")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

if __name__ == '__main__':
    # macOS multiprocessing fix
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # 所有代码已经在模块级别，不需要额外包装
    pass
