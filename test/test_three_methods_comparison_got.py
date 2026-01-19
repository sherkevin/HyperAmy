#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三种检索方法对比实验 - 使用GoT数据集验证泛化能力

方法：
1. hyperamy: 纯情绪检索（使用poincare双曲空间）
2. hipporag: 纯语义检索（标准HippoRAG）
3. fusion: 语义+情绪混合检索（HippoRAGEnhanced，最佳配置: harmonic_none_0.4）

数据集：
- 训练数据: data/processed/got_amygdala.jsonl (1232个chunks)
- QA数据: data/benchmarks/instinct_qa.json (50个QA对)

目标：验证最佳配置在GoT数据集上的泛化能力
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.config import API_KEY, BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
from sentiment.hipporag_enhanced import HippoRAGEnhanced
from sentiment.fusion_strategies import FusionStrategy, NormalizationStrategy
from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from poincare.storage import HyperAmyStorage
from poincare.retrieval import HyperAmyRetrieval
from particle.particle import ParticleEntity
from point_label.emotion import Emotion
from sentence_transformers import SentenceTransformer
import torch

# 设置环境变量
os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['API_KEY'] = API_KEY

# 配置日志
log_file = project_root / 'test_three_methods_comparison_got.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
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
print("三种检索方法对比实验 - GoT数据集（泛化能力验证）")
print("=" * 80)
print("方法说明：")
print("  1. HyperAmy: 纯情绪检索（使用poincare双曲空间）")
print("  2. HippoRAG: 纯语义检索（标准HippoRAG）")
print("  3. Fusion: 语义+情绪混合检索（HippoRAGEnhanced，最佳配置: harmonic_none_0.4）")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 加载数据集
print("\n【步骤1】加载数据集...")
chunks_file = project_root / "data" / "processed" / "got_amygdala.jsonl"
qa_file = project_root / "data" / "benchmarks" / "instinct_qa.json"

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

# 限制测试数量（可选，0表示使用全部）
MAX_QUERIES = 0  # 0=全部50个，也可以改为10进行快速测试
if MAX_QUERIES > 0 and len(qa_pairs) > MAX_QUERIES:
    logger.warning(f"\n⚠️  限制为前 {MAX_QUERIES} 个查询进行测试（总共有 {len(qa_pairs)} 个）")
    qa_pairs = qa_pairs[:MAX_QUERIES]
else:
    logger.info(f"使用全部 {len(qa_pairs)} 个QA对进行测试")

# 准备文档列表（用于索引）
docs = []
chunk_id_to_doc = {}
for chunk in chunks:
    # GoT数据集使用text字段
    text = chunk.get('text') or chunk.get('input') or chunk.get('content') or chunk.get('chunk_text', '')
    # GoT数据集chunk_id是整数，需要转换为字符串格式
    chunk_id_raw = chunk.get('chunk_id')
    if chunk_id_raw is not None:
        chunk_id = f'chunk_{chunk_id_raw}' if isinstance(chunk_id_raw, int) else str(chunk_id_raw)
    else:
        chunk_id = f'chunk_{chunks.index(chunk)}'
    
    # 确保text是字符串且有一定长度
    if isinstance(text, str) and len(text.strip()) > 20:
        docs.append(text.strip())
        chunk_id_to_doc[chunk_id] = text.strip()
    else:
        logger.warning(f"跳过无效chunk (chunk_id={chunk_id}): text长度={len(text) if isinstance(text, str) else 'N/A'}")

logger.info(f"✅ 准备了 {len(docs)} 个文档用于索引")
if len(docs) == 0:
    logger.error("❌ 未能从chunks中提取任何有效文档。请检查数据格式。")
    sys.exit(1)

queries = [qa['question'] for qa in qa_pairs]
logger.info(f"✅ 准备了 {len(queries)} 个查询")

# 准备gold_docs（用于评估）
gold_docs = []
for qa in qa_pairs:
    chunk_id_raw = qa.get('chunk_id')
    if chunk_id_raw is not None:
        # GoT的chunk_id是整数，转换为字符串格式
        chunk_id = f'chunk_{chunk_id_raw}' if isinstance(chunk_id_raw, int) else str(chunk_id_raw)
    else:
        # 如果没有chunk_id，尝试从chunk_text匹配
        chunk_text = qa.get('chunk_text', '')
        chunk_id = None
        for cid, doc_text in chunk_id_to_doc.items():
            if chunk_text.strip() in doc_text or doc_text.strip() in chunk_text:
                chunk_id = cid
                break
    
    gold_text = chunk_id_to_doc.get(chunk_id) if chunk_id else None
    if gold_text:
        gold_docs.append([gold_text])
    else:
        # 如果找不到，使用QA数据中的chunk_text
        chunk_text = qa.get('chunk_text', '')
        if chunk_text:
            gold_docs.append([chunk_text.strip()])
        else:
            gold_docs.append([])

# 创建输出目录
output_dir = project_root / "outputs" / "three_methods_comparison_got"
output_dir.mkdir(parents=True, exist_ok=True)

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

# ========== 方法2: Fusion (语义+情绪混合) ==========
print("\n【步骤5】初始化 HippoRAGEnhanced (Fusion: 语义+情绪混合)...")
save_dir_fusion = output_dir / "fusion"
save_dir_fusion.mkdir(exist_ok=True)

config_fusion = BaseConfig(
    save_dir=str(save_dir_fusion),
    llm_base_url=llm_base_url,
    llm_name=llm_model_name,
    embedding_model_name=embedding_model_name,
    embedding_base_url=embedding_base_url,
    force_index_from_scratch=False,  # 如果索引已存在，复用索引以加速
    retrieval_top_k=5,
)

try:
    fusion = HippoRAGEnhanced(
        global_config=config_fusion,
        enable_sentiment=True,
        sentiment_weight=0.4,  # 最佳权重（基于网格搜索结果）
        sentiment_model_name=llm_model_name,
        max_workers=10,  # 使用并发优化
        fusion_strategy=FusionStrategy.HARMONIC,  # 最佳策略
        normalization_strategy=NormalizationStrategy.NONE  # 最佳归一化
    )
    print("✅ Fusion 初始化成功（最佳配置: harmonic_none_0.4, max_workers=10）")
    
    print("\n【步骤6】索引文档（Fusion）- 使用并发优化...")
    fusion.index(docs=docs)
    print("✅ Fusion 索引完成")
    
    print("\n【步骤7】Fusion 检索...")
    fusion_results, fusion_eval = fusion.retrieve(
        queries=queries,
        num_to_retrieve=5,
        gold_docs=gold_docs
    )
    print("✅ Fusion 检索完成")
    if fusion_eval:
        print(f"   检索评估指标: {fusion_eval}")
    fusion_available = True
except Exception as e:
    logger.error(f"Fusion 失败: {e}")
    import traceback
    traceback.print_exc()
    fusion_available = False
    fusion_results = None
    fusion_eval = None

# ========== 方法3: HyperAmy (纯情绪) ==========
logger.info("\n【步骤8】初始化 HyperAmy (纯情绪检索)...")
try:
    # 使用poincare的HyperAmyRetrieval
    emotion_extractor = Emotion()
    
    # 使用GPU加速SentenceTransformer（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"✅ 使用GPU加速: {torch.cuda.get_device_name(0)}")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # 创建存储
    storage_path = output_dir / "hyperamy_db"
    
    # 检查主存储路径是否已完成（优先使用）
    main_storage_ready = False
    main_id_to_content_file = output_dir / "hyperamy_id_to_content.json"
    id_to_content = {}
    storage = None  # 初始化storage变量
    
    if storage_path.exists() and main_id_to_content_file.exists():
        try:
            with open(main_id_to_content_file, 'r', encoding='utf-8') as f:
                loaded_id_to_content = json.load(f)
            main_storage = HyperAmyStorage(persist_path=str(storage_path))
            main_count = main_storage.collection.count()
            if main_count >= len(loaded_id_to_content) and main_count >= len(chunks) * 0.9:  # 至少90%的点才认为可用
                logger.info(f"✅ 检测到主存储已完成（{main_count}个点），直接使用")
                id_to_content = loaded_id_to_content
                storage = main_storage  # 直接使用main_storage
                main_storage_ready = True
        except Exception as e:
            logger.warning(f"⚠️  读取主存储失败: {e}，将重新索引")
    
    if not main_storage_ready:
        # 如果主存储未准备好，则正常初始化存储并提取情绪向量
        storage = HyperAmyStorage(persist_path=str(storage_path))
        
        # 提取情绪向量并存储点（并发优化）
        print("   提取情绪向量并存储点（并发处理）...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        stored_points = 0
        skipped_count = 0
        max_workers = 15
        logger.info(f"使用并发处理，max_workers={max_workers}")
        
        # 定义并发任务
        def extract_emotion_for_chunk(chunk_data):
            """提取单个chunk的情绪向量（用于并发调用）"""
            chunk_idx, chunk = chunk_data
            # GoT数据集使用text字段
            text = chunk.get('text') or chunk.get('input') or chunk.get('content') or chunk.get('chunk_text', '')
            # GoT数据集chunk_id是整数，转换为字符串格式
            chunk_id_raw = chunk.get('chunk_id')
            if chunk_id_raw is not None:
                chunk_id = f'chunk_{chunk_id_raw}' if isinstance(chunk_id_raw, int) else str(chunk_id_raw)
            else:
                chunk_id = f'chunk_{chunk_idx}'
            
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
                    
                    # 使用chunk_id作为entity_id
                    entity_id = chunk_id
                    weight = float(torch.norm(emotion_vector))
                    
                    return {
                        'success': True,
                        'chunk_idx': chunk_idx,
                        'chunk_id': chunk_id,
                        'entity_id': entity_id,
                        'emotion_vector': emotion_vector,
                        'weight': weight,
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
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(extract_emotion_for_chunk, (chunk_idx, chunk)): chunk_idx
                for chunk_idx, chunk in enumerate(chunks)
            }
            
            results_list = [None] * len(chunks)
            
            for future in tqdm(as_completed(future_to_chunk), 
                            total=len(future_to_chunk), 
                            desc="提取情绪向量"):
                result = future.result()
                results_list[result['chunk_idx']] = result
                
                if result['success']:
                    entity = ParticleEntity(
                        entity_id=result['entity_id'],
                        entity=result['text'][:50],
                        text_id=result['chunk_id'],
                        emotion_vector=result['emotion_vector'].numpy(),
                        weight=result['weight'],
                        speed=0.0,
                        temperature=1.0,
                        born=0.0
                    )
                    storage.upsert_entity(entity)
                    id_to_content[result['entity_id']] = result['text']
                    stored_points += 1
                else:
                    skipped_count += 1
        
        print(f"✅ HyperAmy 存储初始化完成（存储了 {stored_points} 个点，跳过了 {skipped_count} 个无效chunks）")
        
        # 保存id_to_content映射到文件（供后续检索使用）
        with open(main_id_to_content_file, 'w', encoding='utf-8') as f:
            json.dump(id_to_content, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 保存了id->content映射到: {main_id_to_content_file}")
    
    # 创建检索器
    from poincare.physics import ParticleProjector
    projector = ParticleProjector()
    hyperamy_retrieval = HyperAmyRetrieval(storage, projector)
    
    print("\n【步骤9】HyperAmy 检索...")
    hyperamy_results = []
    for query_idx, query in enumerate(tqdm(queries, desc="HyperAmy检索")):
        try:
            # 提取查询的情绪向量
            query_emotion = emotion_extractor.extract(query)
            # 确保emotion_vector是torch.Tensor，并在CPU上
            if isinstance(query_emotion, np.ndarray):
                query_emotion = torch.tensor(query_emotion, dtype=torch.float32)
            elif not isinstance(query_emotion, torch.Tensor):
                query_emotion = torch.tensor(query_emotion, dtype=torch.float32)
            if query_emotion.is_cuda:
                query_emotion = query_emotion.cpu()
            
            # 生成查询的entity_id
            query_id = f"query_{query_idx}_{hashlib.md5(query.encode('utf-8')).hexdigest()[:16]}"
            
            query_entity = ParticleEntity(
                entity_id=query_id,
                entity=query[:50],
                text_id=f"query_{query_idx}",
                emotion_vector=query_emotion.numpy(),
                weight=float(torch.norm(query_emotion)),
                speed=0.0,
                temperature=1.0,
                born=0.0
            )
            # 检索
            search_results = hyperamy_retrieval.search(query_entity, top_k=5)
            # 转换为QuerySolution格式
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
            hyperamy_results.append(QuerySolution(
                question=query,
                docs=[],
                doc_scores=np.array([])
            ))
    logger.info("✅ HyperAmy 检索完成")
    hyperamy_available = True
except Exception as e:
    logger.error(f"HyperAmy 模块初始化或运行失败: {e}")
    import traceback
    traceback.print_exc()
    hyperamy_available = False
    hyperamy_results = [QuerySolution(question=q, docs=[], doc_scores=np.array([])) for q in queries]

# ========== 结果对比和保存 ==========
print("\n【步骤10】结果对比和保存...")

comparison_results = []
for i, qa in enumerate(qa_pairs):
    result = {
        'query_idx': i,
        'question': qa['question'],
        'gold_chunk_id': qa.get('chunk_id'),
        'gold_doc': gold_docs[i][0] if gold_docs[i] else None,
    }
    
    # HippoRAG结果
    if hipporag_available and hipporag_results and i < len(hipporag_results):
        result['hipporag'] = {
            'available': True,
            'docs': hipporag_results[i].docs[:5] if hipporag_results[i] else [],
            'scores': hipporag_results[i].doc_scores.tolist()[:5] if hipporag_results[i] else []
        }
    else:
        result['hipporag'] = {'available': False}
    
    # Fusion结果
    if fusion_available and fusion_results and i < len(fusion_results):
        result['fusion'] = {
            'available': True,
            'docs': fusion_results[i].docs[:5] if fusion_results[i] else [],
            'scores': fusion_results[i].doc_scores.tolist()[:5] if fusion_results[i] else []
        }
    else:
        result['fusion'] = {'available': False}
    
    # HyperAmy结果
    if hyperamy_available and hyperamy_results and i < len(hyperamy_results):
        result['hyperamy'] = {
            'available': True,
            'docs': hyperamy_results[i].docs[:5] if hyperamy_results[i] else [],
            'scores': hyperamy_results[i].doc_scores.tolist()[:5] if hyperamy_results[i] else []
        }
    else:
        result['hyperamy'] = {'available': False}
    
    comparison_results.append(result)

# 保存结果
result_file = output_dir / "comparison_results.json"
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, ensure_ascii=False, indent=2)
logger.info(f"✅ 结果已保存到: {result_file}")

# 评估指标（使用retrieve方法返回的评估结果）
print("\n【步骤11】计算评估指标...")

if hipporag_available and hipporag_eval:
    print(f"\nHippoRAG 评估指标:")
    if isinstance(hipporag_eval, dict):
        print(f"  Recall@1: {hipporag_eval.get('recall_at_1', 0.0):.4f}")
        print(f"  Recall@5: {hipporag_eval.get('recall_at_5', 0.0):.4f}")
        print(f"  MRR: {hipporag_eval.get('mrr', 0.0):.4f}")
    else:
        print(f"  评估结果: {hipporag_eval}")

if fusion_available and fusion_eval:
    print(f"\nFusion 评估指标 (最佳配置: harmonic_none_0.4):")
    if isinstance(fusion_eval, dict):
        print(f"  Recall@1: {fusion_eval.get('recall_at_1', 0.0):.4f}")
        print(f"  Recall@5: {fusion_eval.get('recall_at_5', 0.0):.4f}")
        print(f"  MRR: {fusion_eval.get('mrr', 0.0):.4f}")
    else:
        print(f"  评估结果: {fusion_eval}")

# HyperAmy需要单独评估（因为使用情绪相似度）
if hyperamy_available and hyperamy_results:
    from evaluation.unified_evaluator import UnifiedEvaluator
    evaluator = UnifiedEvaluator()
    hyperamy_metrics = evaluator.evaluate_exact_match(
        method_name="HyperAmy",
        gold_texts=[gd[0] if gd else "" for gd in gold_docs],
        retrieved_texts_list=[hr.docs[:5] if hr else [] for hr in hyperamy_results]
    )
    print(f"\nHyperAmy 评估指标:")
    print(f"  Recall@1: {hyperamy_metrics.recall_at_k.get(1, 0.0):.4f}")
    print(f"  Recall@5: {hyperamy_metrics.recall_at_k.get(5, 0.0):.4f}")
    print(f"  MRR: {hyperamy_metrics.mrr:.4f}")

print("\n" + "=" * 80)
print("✅ 实验完成！")
print("=" * 80)
print(f"结果文件: {result_file}")
print("=" * 80)

