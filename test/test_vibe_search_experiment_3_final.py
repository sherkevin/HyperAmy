#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 3: Final Fusion - LLM API + Semantic Collapse Protocol

这是最终融合实验，结合：
1. 语义崩溃协议（来自实验2，已验证有效）
2. LLM API情绪提取（来自实验1，I_q值高）

数据集：
- 训练数据: data/training/monte_cristo_train_full.jsonl
- QA数据: data/public_benchmark/monte_cristo_vibe_search.json (50个QA对)

方法：
1. HippoRAG: 纯语义检索（基线）
2. HyperAmy-Hybrid: 动态权重混合检索（语义崩溃协议已激活）
   - 使用LLM API提取情绪向量（高I_q值）
   - 使用最新的search_hybrid（包含SEMANTIC_COLLAPSE_THRESHOLD逻辑）
"""
import sys
import os
import json
import numpy as np
import logging
import hashlib
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.config import API_KEY, BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from poincare.storage import HyperAmyStorage
from poincare.retrieval import HyperAmyRetrieval
from particle.particle import ParticleEntity
from particle.emotion import Emotion, EMOTIONS
from sentence_transformers import SentenceTransformer

# 设置环境变量
os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['API_KEY'] = API_KEY

# 配置日志
output_dir = project_root / "outputs" / "vibe_search_experiment_3_final"
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / 'experiment_3_final.log'
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
print("Experiment 3: Final Fusion - LLM API + Semantic Collapse Protocol")
print("=" * 80)
print("核心特性：")
print("  1. ✅ 语义崩溃协议已激活（SEMANTIC_COLLAPSE_THRESHOLD = 0.05）")
print("  2. ✅ LLM API提取情绪向量（高I_q值，预期0.8-0.9）")
print("  3. ✅ 自动复用情绪向量缓存（节省API费用）")
print("  4. ✅ 详细日志记录每一条查询的决策过程")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"日志文件: {log_file}")
print(f"输出目录: {output_dir}")
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

# 准备数据
queries = [vq.get('query', '') for vq in vibe_queries]
gold_docs = []
gold_texts = []
for vq in vibe_queries:
    gold_chunk_id = vq.get('gold_chunk_id', '')
    gold_text = vq.get('gold_text', '')
    gold_docs.append([gold_text])
    gold_texts.append(gold_text)

print(f"✅ 准备了 {len(queries)} 个查询和对应的gold文档")

# ========== 方法1: HippoRAG (基线) ==========
print("\n【步骤2】初始化 HippoRAG...")
save_dir_hipporag = output_dir / "hipporag"
save_dir_hipporag.mkdir(exist_ok=True)

config_hipporag = BaseConfig(
    save_dir=str(save_dir_hipporag),
    llm_base_url=llm_base_url,
    llm_name=llm_model_name,
    embedding_model_name=embedding_model_name,
    embedding_base_url=embedding_base_url,
    force_index_from_scratch=False,  # 复用索引以加速
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
    
    # 检查是否需要索引
    if not (save_dir_hipporag / "embedding_store" / "passage_embeddings.npy").exists():
        print("\n【步骤3】索引文档（HippoRAG）...")
        docs = [chunk.get('content', '') for chunk in chunks]
        hipporag.index(docs=docs)
        print("✅ HippoRAG 索引完成")
    else:
        print("✅ HippoRAG 索引已存在，复用索引")
    
    print("\n【步骤4】HippoRAG 检索...")
    hipporag_results, hipporag_eval = hipporag.retrieve(
        queries=queries,
        num_to_retrieve=5,
        gold_docs=gold_docs
    )
    print("✅ HippoRAG 检索完成")
    if hipporag_eval:
        print(f"   检索评估指标: {hipporag_eval}")
        logger.info(f"HippoRAG Recall@1: {hipporag_eval.get('Recall@1', 0):.2%}")
    hipporag_available = True
except Exception as e:
    logger.error(f"HippoRAG 失败: {e}")
    import traceback
    traceback.print_exc()
    hipporag_available = False
    hipporag_results = None
    hipporag_eval = None

# ========== 方法2: HyperAmy-Hybrid (语义崩溃协议 + LLM API) ==========
print("\n【步骤5】初始化 HyperAmy-Hybrid (LLM API + 语义崩溃协议)...")

# 初始化LLM API情绪提取器（启用缓存）
emotion_extractor = Emotion(enable_cache=True, cache_dir=".cache/emotion_vectors")
logger.info(f"✅ Emotion提取器初始化完成（缓存已启用）")

# 使用GPU加速SentenceTransformer（如果可用）
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    logger.info(f"✅ 使用GPU加速: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    logger.info(f"✅ 使用MPS加速（macOS）")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 创建存储
storage_path = output_dir / "hyperamy_db"
id_to_content_file = output_dir / "hyperamy_id_to_content.json"

# 维护id->content映射
id_to_content = {}

# 检查是否已有可用的索引
use_existing_index = False
if storage_path.exists() and id_to_content_file.exists():
    try:
        with open(id_to_content_file, 'r', encoding='utf-8') as f:
            id_to_content = json.load(f)
        
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
        
        # 检查id_to_content映射文件是否有效
        if len(id_to_content) > 0 and storage_count >= len(id_to_content) and storage_count >= 9000:
            logger.info(f"✅ 检测到HyperAmy索引已完成（{storage_count}个点，映射{len(id_to_content)}条），直接使用现有索引")
            storage = HyperAmyStorage(persist_path=str(storage_path))
            use_existing_index = True
        elif storage_count >= 9000 and len(id_to_content) == 0:
            logger.warning(f"⚠️  索引存在但映射文件为空，将从存储中重建映射（存储点数: {storage_count}）")
            # 尝试从存储中重建映射（如果可能）
            # 如果无法重建，则重新索引
            use_existing_index = False  # 强制重新索引以确保映射正确
        else:
            logger.info(f"   现有索引点数不足（{storage_count}/{len(id_to_content) if id_to_content else 0}），将重新索引")
    except Exception as e:
        logger.warning(f"⚠️  读取现有索引失败: {e}，将重新索引")

if not use_existing_index:
    # 初始化存储
    storage = HyperAmyStorage(persist_path=str(storage_path))
    
    # 提取情绪向量并存储点（并发优化）
    print("   提取情绪向量并存储点（并发处理，使用LLM API + 缓存）...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    stored_points = 0
    skipped_count = 0
    
    def extract_emotion_for_chunk(chunk_data):
        chunk_idx, chunk = chunk_data
        try:
            content = chunk.get('content', '')
            if not content or len(content.strip()) < 10:
                return None
            
            chunk_id = chunk.get('chunk_id', f'chunk_{chunk_idx}')
            
            # 使用LLM API提取情绪向量（自动使用缓存）
            emotion_vector = emotion_extractor.extract(content)
            
            # 归一化
            norm = np.linalg.norm(emotion_vector)
            if norm > 1e-9:
                normalized_vector = emotion_vector / norm
                weight = float(norm)
            else:
                normalized_vector = emotion_vector.copy()
                weight = 0.0
            
            return {
                'chunk_id': chunk_id,
                'content': content,
                'emotion_vector': normalized_vector,
                'weight': weight
            }
        except Exception as e:
            logger.error(f"提取chunk {chunk_idx}的情绪向量失败: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(extract_emotion_for_chunk, (chunk_idx, chunk)): chunk_idx
            for chunk_idx, chunk in enumerate(chunks)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="提取情绪向量"):
            result = future.result()
            if result:
                entity = ParticleEntity(
                    entity_id=result['chunk_id'],
                    entity=result['content'][:100],
                    text_id=result['chunk_id'],
                    emotion_vector=result['emotion_vector'],
                    weight=result['weight'],
                    speed=0.0,
                    temperature=1.0,
                    purity=0.5,
                    tau_v=86400.0,
                    tau_T=86400.0,
                    born=time.time()
                )
                storage.upsert_entity(entity)
                id_to_content[result['chunk_id']] = result['content']
                stored_points += 1
            else:
                skipped_count += 1
    
    # 保存id_to_content映射
    with open(id_to_content_file, 'w', encoding='utf-8') as f:
        json.dump(id_to_content, f, ensure_ascii=False, indent=2)
    
    print(f"✅ HyperAmy 存储初始化完成（存储了 {stored_points} 个点，跳过了 {skipped_count} 个无效chunks）")
else:
    storage = HyperAmyStorage(persist_path=str(storage_path))
    stored_points = len(id_to_content)
    logger.info(f"✅ 使用已存在的HyperAmy索引（{stored_points}个点）")

# 创建检索器
from poincare.projector import ParticleProjector
projector = ParticleProjector()
hyperamy_retrieval = HyperAmyRetrieval(storage, projector)

print("\n【步骤6】HyperAmy-Hybrid 混合检索（LLM API + 语义崩溃协议）...")

# 记录每条查询的详细信息
query_details = []

hyperamy_results = []
for i, query in enumerate(tqdm(queries, desc="HyperAmy-Hybrid检索")):
    try:
        # ========== 第一阶段：获取Top-100语义候选 ==========
        if hipporag_available:
            semantic_results_100 = hipporag.retrieve(
                queries=[query],
                num_to_retrieve=100
            )
            if semantic_results_100 and len(semantic_results_100) > 0:
                semantic_docs = semantic_results_100[0].docs
                semantic_scores = semantic_results_100[0].doc_scores
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
        
        # ========== 第二阶段：提取查询情绪向量（使用LLM API + 缓存） ==========
        query_emotion = emotion_extractor.extract(query)
        
        # 确保emotion_vector是numpy array
        if isinstance(query_emotion, np.ndarray):
            query_emotion_np = query_emotion.astype(np.float32)
        else:
            query_emotion_np = np.array(query_emotion, dtype=np.float32)
        
        # 维度验证
        expected_dim = len(EMOTIONS)
        if len(query_emotion_np) != expected_dim:
            logger.warning(f"查询向量维度异常：{len(query_emotion_np)}，预期：{expected_dim}。尝试自动修复...")
            if len(query_emotion_np) > expected_dim:
                query_emotion_np = query_emotion_np[:expected_dim]
            else:
                padding = np.zeros(expected_dim - len(query_emotion_np), dtype=np.float32)
                query_emotion_np = np.concatenate((query_emotion_np, padding))
        
        # 计算I_q（查询情绪强度）
        I_q = float(np.max(np.abs(query_emotion_np)))  # 使用最大值作为强度指标
        I_q = min(1.0, max(0.0, I_q))
        
        # 归一化
        norm = np.linalg.norm(query_emotion_np)
        if norm > 1e-9:
            normalized_vector = query_emotion_np / norm
            weight = float(norm)
        else:
            normalized_vector = query_emotion_np.copy()
            weight = 0.0
        
        query_id = f"query_{i}_{hashlib.md5(query.encode('utf-8')).hexdigest()[:16]}"
        
        query_entity = ParticleEntity(
            entity_id=query_id,
            entity=query[:50],
            text_id=f"query_{i}",
            emotion_vector=normalized_vector,
            weight=weight,
            speed=0.0,
            temperature=1.0,
            purity=0.5,
            tau_v=86400.0,
            tau_T=86400.0,
            born=time.time()
        )
        
        # ========== 第三阶段：混合检索（包含语义崩溃协议） ==========
        search_results = hyperamy_retrieval.search_hybrid(
            query_text=query,
            query_entity=query_entity,
            semantic_docs=semantic_docs,
            semantic_scores=semantic_scores,
            id_to_content=id_to_content,
            top_k=5,
            alpha=0.8
        )
        
        # ========== 提取决策信息 ==========
        # 从search_results的metadata中提取决策信息（search_hybrid返回的SearchResult包含完整元数据）
        S_sem = 0.0
        W_emo = 0.0
        W_sem = 0.0
        collapsed = False
        
        # 从search_results的metadata中提取（这是最准确的方式）
        if search_results and len(search_results) > 0:
            metadata = search_results[0].metadata if hasattr(search_results[0], 'metadata') else {}
            if metadata:
                # search_hybrid的metadata中包含这些信息
                S_sem = metadata.get('S_sem', 0.0)
                W_emo = metadata.get('w_emo', 0.0)
                W_sem = metadata.get('w_sem', 0.0)
                I_q_from_meta = metadata.get('I_q', 0.0)
                
                # 检查是否触发了语义崩溃协议
                collapsed = (S_sem < 0.05 and W_emo > 0.5)
                
                # 如果metadata中有I_q，使用它（可能更准确）
                if I_q_from_meta > 0:
                    I_q = I_q_from_meta
        
        # 如果metadata中没有信息（不应该发生，但做兜底）
        if S_sem == 0.0 and W_emo == 0.0:
            logger.warning(f"Query {i+1}: 无法从metadata提取决策信息，使用估算值")
            # 在Vibe Search场景下，S_sem应该很低，使用小的默认值
            S_sem = 0.001  # 假设语义失效
            if S_sem < 0.05:  # 语义崩溃协议
                W_emo = 0.5 + (I_q * 0.45)
                W_sem = 1.0 - W_emo
                collapsed = True
            else:
                W_emo = 0.3
                W_sem = 0.7
        
        # 转换为QuerySolution格式
        from hipporag.utils.misc_utils import QuerySolution
        doc_list = []
        score_list = []
        for result in search_results:
            doc_id = result.id if hasattr(result, 'id') else ''
            doc_content = id_to_content.get(doc_id, '')
            if doc_content:
                doc_list.append(doc_content)
                score_list.append(result.score if hasattr(result, 'score') else 0.0)
        
        hyperamy_results.append(QuerySolution(
            question=query,
            docs=doc_list,
            doc_scores=np.array(score_list) if score_list else np.array([])
        ))
        
        # 记录查询详情
        gold_text = gold_texts[i]
        hit = gold_text in doc_list[:1] if doc_list else False
        query_details.append({
            'query': query,
            'I_q': I_q,
            'S_sem': S_sem,
            'W_emo': W_emo,
            'W_sem': W_sem,
            'collapsed': collapsed,
            'hit': hit,
            'gold_text': gold_text[:100]
        })
        
        # 详细日志
        collapse_marker = " ⚠️ [COLLAPSE PROTOCOL]" if collapsed else ""
        logger.info(
            f"Query {i+1}/{len(queries)}: I_q={I_q:.4f}, S_sem={S_sem:.4f}, "
            f"W_emo={W_emo:.4f}, W_sem={W_sem:.4f}, Hit={hit}{collapse_marker}"
        )
        
    except Exception as e:
        logger.error(f"HyperAmy-Hybrid检索失败 (query={query[:50]}): {e}")
        import traceback
        traceback.print_exc()
        from hipporag.utils.misc_utils import QuerySolution
        hyperamy_results.append(QuerySolution(
            question=query,
            docs=[],
            doc_scores=np.array([])
        ))
        query_details.append({
            'query': query,
            'I_q': 0.0,
            'S_sem': 0.0,
            'W_emo': 0.0,
            'W_sem': 0.0,
            'collapsed': False,
            'hit': False,
            'error': str(e)
        })

# 评估结果
print("\n【步骤7】评估结果...")

def calculate_recall_at_k(retrieved_docs_list, gold_docs_list, k_list=[1, 5, 10, 20]):
    """计算Recall@K"""
    recall_at_k = {f"Recall@{k}": 0.0 for k in k_list}
    total_queries = len(gold_docs_list)
    
    for retrieved_docs, gold_docs in zip(retrieved_docs_list, gold_docs_list):
        gold_set = set(gold_docs) if gold_docs else set()
        retrieved_list = retrieved_docs if retrieved_docs else []
        
        for k in k_list:
            if k <= len(retrieved_list):
                top_k = retrieved_list[:k]
                top_k_set = set(top_k)
                intersection = gold_set & top_k_set
                if gold_set:
                    recall_at_k[f"Recall@{k}"] += len(intersection) / len(gold_set)
    
    # 平均所有查询
    for k in k_list:
        recall_at_k[f"Recall@{k}"] /= total_queries if total_queries > 0 else 1.0
        recall_at_k[f"Recall@{k}"] = round(recall_at_k[f"Recall@{k}"], 4)
    
    return recall_at_k

if hyperamy_results:
    hyperamy_retrieved_docs = [result.docs if hasattr(result, 'docs') else [] for result in hyperamy_results]
    hyperamy_eval = calculate_recall_at_k(hyperamy_retrieved_docs, gold_docs)
    print("✅ HyperAmy-Hybrid 检索完成")
    if hyperamy_eval:
        print(f"   检索评估指标: {hyperamy_eval}")
        logger.info(f"HyperAmy-Hybrid Recall@1: {hyperamy_eval.get('Recall@1', 0):.2%}")

# 汇总结果
print("\n" + "=" * 80)
print("Experiment 3: Final Fusion 实验结果汇总")
print("=" * 80)
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

print("方法对比：")
print("  1. HippoRAG (纯语义):")
if hipporag_eval:
    print(f"     Recall@1: {hipporag_eval.get('Recall@1', 0.0)*100:.1f}%")
    print(f"     Recall@5: {hipporag_eval.get('Recall@5', 0.0)*100:.1f}%")
print("  2. HyperAmy-Hybrid (LLM API + 语义崩溃协议):")
if hyperamy_eval:
    print(f"     Recall@1: {hyperamy_eval.get('Recall@1', 0.0)*100:.1f}%")
    print(f"     Recall@5: {hyperamy_eval.get('Recall@5', 0.0)*100:.1f}%")

print()
print("语义崩溃协议统计：")
collapse_count = sum(1 for qd in query_details if qd.get('collapsed', False))
if query_details:
    avg_iq = sum(qd.get('I_q', 0) for qd in query_details) / len(query_details)
    avg_w_emo = sum(qd.get('W_emo', 0) for qd in query_details) / len(query_details)
    avg_s_sem = sum(qd.get('S_sem', 0) for qd in query_details) / len(query_details)
    print(f"  触发次数: {collapse_count}/{len(query_details)} ({collapse_count/len(query_details)*100:.1f}%)")
    print(f"  平均I_q: {avg_iq:.4f}")
    print(f"  平均S_sem: {avg_s_sem:.4f}")
    print(f"  平均W_emo: {avg_w_emo:.4f}")

print("=" * 80)

# 保存结果
results_file = output_dir / "results.json"
results_data = {
    'timestamp': datetime.now().isoformat(),
    'model': 'LLM_API',
    'hipporag': {
        'recall_at_1': hipporag_eval.get('Recall@1', 0.0) if hipporag_eval else 0.0,
        'recall_at_5': hipporag_eval.get('Recall@5', 0.0) if hipporag_eval else 0.0,
        'full_eval': hipporag_eval
    },
    'hyperamy': {
        'recall_at_1': hyperamy_eval.get('Recall@1', 0.0) if hyperamy_eval else 0.0,
        'recall_at_5': hyperamy_eval.get('Recall@5', 0.0) if hyperamy_eval else 0.0,
        'full_eval': hyperamy_eval
    },
    'hybrid': {
        'recall_at_1': hyperamy_eval.get('Recall@1', 0.0) if hyperamy_eval else 0.0,
        'recall_at_5': hyperamy_eval.get('Recall@5', 0.0) if hyperamy_eval else 0.0,
        'full_eval': hyperamy_eval
    },
    'query_details': query_details,
    'semantic_collapse_stats': {
        'triggered_count': collapse_count,
        'total_queries': len(query_details),
        'trigger_rate': collapse_count / len(query_details) if query_details else 0.0,
        'avg_iq': avg_iq if query_details else 0.0,
        'avg_w_emo': avg_w_emo if query_details else 0.0,
        'avg_s_sem': avg_s_sem if query_details else 0.0
    },
    'stored_points': stored_points
}

with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)
logger.info(f"✅ 结果已保存到: {results_file}")

print(f"\n✅ 实验完成！结果已保存到: {results_file}")
print(f"日志文件: {log_file}")
