#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vibe Search / Emotional Retrieval 实验 - 使用Emos模型 + GPU加速

这是 test_vibe_search_experiment_final.py 的并行版本，使用：
- EmosWrapper（本地模型）代替Emotion（LLM API）
- MPS GPU加速（macOS）
- 独立的输出目录和日志文件，不影响正在运行的实验

方法：
1. HippoRAG: 纯语义检索（标准HippoRAG，复用现有索引）
2. HyperAmy (Emos): 纯情绪检索（使用poincare双曲空间，Emos模型抽取情绪向量）
3. Hybrid (Dynamic v2): 动态权重混合检索（Weighted RRF + Adaptive Weighting）

数据集：
- 训练数据: data/training/monte_cristo_train_full.jsonl
- QA数据: data/public_benchmark/monte_cristo_vibe_search.json (50个QA对)
"""
import sys
import os
import json
import numpy as np
import logging
import hashlib
import uuid
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# 设置HuggingFace镜像环境变量（优先使用本地缓存，如果网络有问题则尝试镜像）
if "HF_ENDPOINT" not in os.environ:
    # 检查本地缓存是否存在
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    qwen_cache = cache_dir / "models--Qwen--Qwen3-Embedding-8B"
    if not qwen_cache.exists() or not any(qwen_cache.iterdir()):
        # 如果本地缓存不存在或为空，尝试使用HF镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"⚠️  未找到本地缓存，使用HF镜像: {os.environ['HF_ENDPOINT']}")
    else:
        print(f"✅ 找到本地缓存: {qwen_cache}")
        # 即使有本地缓存，也设置镜像作为备选
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"✅ 设置HF镜像作为备选: {os.environ['HF_ENDPOINT']}")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# macOS多进程支持
multiprocessing.set_start_method('spawn', force=True)

from llm.config import API_KEY, BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from poincare.storage import HyperAmyStorage
from poincare.retrieval import HyperAmyRetrieval
from particle.particle import ParticleEntity
from sentence_transformers import SentenceTransformer
import torch

# 设置环境变量
os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['API_KEY'] = API_KEY

# 配置日志（使用独立的日志文件）
log_file = project_root / 'test_vibe_search_experiment_emos_gpu.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入EmosWrapper，处理可能的导入错误（在logger定义之后）
try:
    from particle.emos_wrapper import EmosWrapper, EMOS_AVAILABLE
    logger.info(f"✅ EmosWrapper导入成功，EMOS_AVAILABLE={EMOS_AVAILABLE}")
except ImportError as e:
    logger.error(f"❌ 无法导入EmosWrapper: {e}")
    logger.error("请确保emos-master目录存在，或设置EMOS_PATH环境变量")
    EMOS_AVAILABLE = False
    EmosWrapper = None
except Exception as e:
    logger.error(f"❌ 导入EmosWrapper时发生错误: {e}")
    import traceback
    traceback.print_exc()
    EMOS_AVAILABLE = False
    EmosWrapper = None

# 模型配置
llm_model_name = DEFAULT_MODEL
llm_base_url = BASE_URL
embedding_model_name = f"VLLM/{DEFAULT_EMBEDDING_MODEL}"
embedding_base_url = API_URL_EMBEDDINGS

# GPU配置（macOS使用MPS）
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"使用设备: {device} (MPS可用: {torch.backends.mps.is_available()})")

print("=" * 80)
print("Vibe Search / Emotional Retrieval 实验 - Emos模型 + GPU加速")
print("=" * 80)
print("方法说明：")
print("  1. HippoRAG: 纯语义检索（标准HippoRAG，复用现有索引）")
print("  2. HyperAmy (Emos): 纯情绪检索（使用poincare双曲空间，Emos模型抽取情绪向量）")
print("  3. Hybrid (Dynamic v2): 动态权重混合检索（Weighted RRF + Adaptive Weighting）")
print("=" * 80)
print(f"设备: {device}")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 检查Emos模型可用性
if not EMOS_AVAILABLE or EmosWrapper is None:
    logger.error("❌ Emos模型不可用，请检查导入路径")
    logger.error("   需要确保emos-master目录存在，或设置EMOS_PATH环境变量")
    logger.error("   脚本无法继续运行，退出")
    sys.exit(1)

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

# 转换为qa_pairs格式
qa_pairs = []
for vq in vibe_queries:
    qa_pairs.append({
        'question': vq.get('query', ''),
        'chunk_id': vq.get('gold_chunk_id', ''),
        'answer': '',
        'gold_text': vq.get('gold_text', ''),
        'emotion_tag': vq.get('emotion_tag', ''),
        'emotion_intensity': vq.get('emotion_intensity', 0.0),
        'requires_emotional_sensitivity': True
    })
print(f"✅ 转换了 {len(qa_pairs)} 个QA对")

# 准备文档列表（用于索引）
docs = []
chunk_id_to_doc = {}
for chunk_idx, chunk in enumerate(chunks):
    text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
    chunk_id = chunk.get('chunk_id') or chunk.get('id') or f'chunk_{chunk_idx}'
    
    if isinstance(text, str) and len(text.strip()) > 20:
        docs.append(text.strip())
        chunk_id_to_doc[chunk_id] = text.strip()

print(f"✅ 准备了 {len(docs)} 个文档用于索引")

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

# 创建输出目录（使用独立的目录）
output_dir = project_root / "outputs" / "vibe_search_experiment_emos_gpu"
output_dir.mkdir(parents=True, exist_ok=True)

results = []

# ========== 方法1: HippoRAG (纯语义) - 复用现有索引 ==========
print("\n【步骤2】初始化 HippoRAG (复用现有索引)...")
save_dir_hipporag = project_root / "outputs" / "vibe_search_experiment" / "hipporag"

config_hipporag = BaseConfig(
    save_dir=str(save_dir_hipporag),
    llm_base_url=llm_base_url,
    llm_name=llm_model_name,
    embedding_model_name=embedding_model_name,
    embedding_base_url=embedding_base_url,
    force_index_from_scratch=False,  # 复用现有索引
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
    print("✅ HippoRAG 初始化成功（复用现有索引）")
    
    # 直接检索（索引应该已经存在）
    print("\n【步骤3】HippoRAG 检索...")
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

# ========== 方法2: HyperAmy (Emos模型) ==========
print("\n【步骤4】初始化 HyperAmy (Emos模型 + GPU加速)...")

# 初始化EmosWrapper
emos_checkpoint_path = project_root / "outputs" / "stage2_training_remote" / "checkpoints" / "best_model_stage2.pt"
if not emos_checkpoint_path.exists():
    logger.warning(f"Emos checkpoint不存在: {emos_checkpoint_path}")
    logger.warning("尝试使用默认路径...")
    emos_checkpoint_path = None  # 让EmosWrapper使用默认路径

try:
    # 尝试使用last_checkpoint.pt（包含配置信息），如果不存在再使用best_model_stage2.pt
    emos_checkpoint_path_last = project_root / "outputs/stage2_training_remote/checkpoints/last_checkpoint.pt"
    if emos_checkpoint_path_last.exists():
        emos_checkpoint_path = emos_checkpoint_path_last
        logger.info(f"使用last_checkpoint.pt: {emos_checkpoint_path}")
        model_name = None  # last_checkpoint.pt包含配置，不需要model_name
    else:
        # 使用best_model_stage2.pt时需要提供model_name
        # 根据checkpoint的state_dict键名分析，best_model_stage2.pt是用RoBERTa-base训练的
        # （键名包含backbone.embeddings.word_embeddings，这是RoBERTa/BERT架构的特征）
        # 同时，checkpoint的embedding_dim=64（从semantic_head维度推断）
        model_name = "roberta-base"
        logger.info(f"使用best_model_stage2.pt，model_name={model_name} (RoBERTa-base, embedding_dim=64)")
    
    # 注意：EmosWrapper会自动从checkpoint推断embedding_dim，但如果checkpoint没有配置信息，
    # 需要确保代码使用正确的默认值。由于best_model_stage2.pt没有配置信息，我们需要
    # 确保EmosWrapper使用正确的参数。
    # 检查checkpoint是否有配置信息
    import torch
    checkpoint_data = torch.load(str(emos_checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint_data, dict) and "config" in checkpoint_data:
        logger.info(f"Checkpoint包含配置信息，将自动使用正确的参数")
    else:
        logger.warning(f"Checkpoint不包含配置信息，将使用默认参数（可能需要手动指定embedding_dim=64）")
    
    emos_wrapper = EmosWrapper(
        checkpoint_path=str(emos_checkpoint_path) if emos_checkpoint_path else None,
        device=device,  # 使用MPS GPU
        model_name=model_name if 'model_name' in locals() else None,
        use_8bit_quantization=False  # 根据实际情况调整
    )
    logger.info(f"✅ EmosWrapper初始化成功 (device: {device})")
except Exception as e:
    logger.error(f"❌ EmosWrapper初始化失败: {e}")
    import traceback
    traceback.print_exc()
    print("⚠️  Emos模型初始化失败，无法继续实验")
    sys.exit(1)

# 使用GPU加速SentenceTransformer
embedding_device = "mps" if torch.backends.mps.is_available() else "cpu"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=embedding_device)
logger.info(f"✅ SentenceTransformer使用设备: {embedding_device}")

# 创建存储（使用独立的目录）
storage_path = output_dir / "hyperamy_db_emos"
id_to_content_file = output_dir / "hyperamy_id_to_content_emos.json"

id_to_content = {}

# 检查是否已有可用的索引
use_existing_index = False
if storage_path.exists() and id_to_content_file.exists():
    try:
        with open(id_to_content_file, 'r', encoding='utf-8') as f:
            id_to_content = json.load(f)
        
        temp_storage = HyperAmyStorage(persist_path=str(storage_path))
        storage_count = temp_storage.collection.count()
        
        if storage_count >= len(id_to_content) and storage_count >= 9000:
            logger.info(f"✅ 检测到HyperAmy索引已完成（{storage_count}个点），直接使用现有索引")
            storage = HyperAmyStorage(persist_path=str(storage_path))
            use_existing_index = True
        else:
            logger.info(f"   现有索引点数不足（{storage_count}/{len(id_to_content)}），将重新索引")
    except Exception as e:
        logger.warning(f"⚠️  读取现有索引失败: {e}，将重新索引")

if not use_existing_index:
    storage = HyperAmyStorage(persist_path=str(storage_path))
    
    print("   提取情绪向量并存储点（Emos模型 + GPU加速，并发处理）...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def extract_emotion_for_chunk_emos(chunk_data):
        """使用Emos模型提取单个chunk的情绪向量"""
        chunk_idx, chunk = chunk_data
        text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
        chunk_id = chunk.get('chunk_id') or chunk.get('id') or f'chunk_{chunk_idx}'
        
        if not isinstance(text, str) or len(text.strip()) < 20:
            return {
                'success': False,
                'chunk_idx': chunk_idx,
                'chunk_id': chunk_id,
                'error': f"text长度不足或无效"
            }
        
        try:
            # 对于chunk级别，将整个chunk作为一个"实体"传递给emos模型
            # 提取前100个字符作为"实体"文本（emos模型需要entity_text）
            # 或者使用整个chunk的前50%作为实体
            entity_text = text[:min(200, len(text) // 2)] if len(text) > 200 else text
            
            # 使用EmosWrapper提取情绪向量
            emotion_vector = emos_wrapper.extract_entity_emotion_vector(
                text=text,
                entity_text=entity_text
            )
            
            # 确保emotion_vector是numpy array
            if isinstance(emotion_vector, torch.Tensor):
                emotion_vector = emotion_vector.cpu().numpy()
            elif not isinstance(emotion_vector, np.ndarray):
                emotion_vector = np.array(emotion_vector, dtype=np.float32)
            
            # 计算weight（原始情绪向量的模长）
            weight = float(np.linalg.norm(emotion_vector))
            
            # 归一化情绪向量
            if weight > 1e-9:
                normalized_vector = emotion_vector / weight
            else:
                normalized_vector = emotion_vector.copy()
                weight = 0.0
            
            entity_id = chunk_id
            
            return {
                'success': True,
                'chunk_idx': chunk_idx,
                'chunk_id': chunk_id,
                'entity_id': entity_id,
                'emotion_vector': normalized_vector,
                'weight': weight,
                'text': text.strip(),
                'error': None
            }
        except Exception as e:
            logger.warning(f"处理chunk失败 (索引={chunk_idx}, chunk_id={chunk_id}): {e}")
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
    
    # 使用ThreadPoolExecutor并发处理（GPU加速，可以适当增加并发数）
    max_workers = 5  # Emos模型是本地GPU推理，并发数可以稍微低一些
    stored_points = 0
    skipped_count = 0
    logger.info(f"使用并发处理（Emos模型 + GPU），max_workers={max_workers}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(extract_emotion_for_chunk_emos, (chunk_idx, chunk)): chunk_idx
            for chunk_idx, chunk in enumerate(chunks)
        }
        
        results_list = []
        for future in tqdm(as_completed(future_to_chunk), 
                          total=len(chunks), 
                          desc="提取情绪向量（Emos+GPU）"):
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
                id_to_content[result['entity_id']] = result['text']
                
                entity = ParticleEntity(
                    entity_id=result['entity_id'],
                    entity=f"chunk_{result['chunk_idx']}",
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
                stored_points += 1
            else:
                skipped_count += 1
        
        # 保存id_to_content映射
        with open(id_to_content_file, 'w', encoding='utf-8') as f:
            json.dump(id_to_content, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 保存了id->content映射到: {id_to_content_file}")
        
        print(f"✅ HyperAmy (Emos) 存储初始化完成（存储了 {stored_points} 个点，跳过了 {skipped_count} 个无效chunks）")
else:
    stored_points = len(id_to_content)
    logger.info(f"✅ 使用并行索引结果（{stored_points}个点）")

# 创建检索器
from poincare.projector import ParticleProjector
projector = ParticleProjector()
hyperamy_retrieval = HyperAmyRetrieval(storage, projector)

print("\n【步骤5】HyperAmy-Hybrid 混合检索（语义+情绪重排序，Emos版本）...")

# 对于Emos模型，我们需要一个适配器来提取查询的情绪向量
# 将整个查询作为"实体"传递给emos模型
def extract_query_emotion_emos(query_text: str) -> np.ndarray:
    """使用Emos模型提取查询的情绪向量"""
    try:
        # 将查询作为"实体"处理
        entity_text = query_text[:min(200, len(query_text))] if len(query_text) > 200 else query_text
        
        emotion_vector = emos_wrapper.extract_entity_emotion_vector(
            text=query_text,
            entity_text=entity_text
        )
        
        # 确保是numpy array
        if isinstance(emotion_vector, torch.Tensor):
            emotion_vector = emotion_vector.cpu().numpy()
        elif not isinstance(emotion_vector, np.ndarray):
            emotion_vector = np.array(emotion_vector, dtype=np.float32)
        
        return emotion_vector.astype(np.float32)
    except Exception as e:
        logger.error(f"提取查询情绪向量失败: {e}")
        # 返回零向量作为fallback（维度需要与存储的向量一致）
        # 假设emos模型返回256维向量
        return np.zeros(256, dtype=np.float32)

hyperamy_results = []
for i, query in enumerate(tqdm(queries, desc="HyperAmy-Hybrid检索（Emos）")):
    try:
        # 第一阶段：获取Top-100语义候选
        if hipporag_available:
            semantic_results_100 = hipporag.retrieve(
                queries=[query],
                num_to_retrieve=100
            )
            if semantic_results_100 and len(semantic_results_100) > 0:
                semantic_docs = semantic_results_100[0].docs
                semantic_scores = semantic_results_100[0].doc_scores
            else:
                logger.warning(f"HyperAmy-Hybrid: 无法获取语义候选，跳过")
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
        
        # 第二阶段：提取查询情绪向量（使用Emos模型）
        query_emotion = extract_query_emotion_emos(query)
        
        # 归一化
        weight = float(np.linalg.norm(query_emotion))
        if weight > 1e-9:
            normalized_vector = query_emotion / weight
        else:
            normalized_vector = query_emotion.copy()
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
        
        # 第三阶段：混合检索
        search_results = hyperamy_retrieval.search_hybrid(
            query_text=query,
            query_entity=query_entity,
            semantic_docs=semantic_docs,
            semantic_scores=semantic_scores,
            id_to_content=id_to_content,
            top_k=5,
            alpha=0.8
        )
        
        # 转换为QuerySolution格式
        from hipporag.utils.misc_utils import QuerySolution
        doc_list = []
        score_list = []
        for result in search_results:
            # SearchResult 是对象，不是字典，使用属性访问
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

# 评估结果
print("\n【步骤6】评估结果...")
# 手动计算Recall@K（因为evaluate_retrieval_results可能不存在）
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
    # 提取检索到的文档列表
    hyperamy_retrieved_docs = [result.docs if hasattr(result, 'docs') else [] for result in hyperamy_results]
    hyperamy_eval = calculate_recall_at_k(hyperamy_retrieved_docs, gold_docs)
    print("✅ HyperAmy (Emos) 检索完成")
    if hyperamy_eval:
        print(f"   检索评估指标: {hyperamy_eval}")

# 汇总结果
print("\n" + "=" * 80)
print("实验结果汇总（Emos模型 + GPU加速）")
print("=" * 80)
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("")
print("方法对比：")
print("  1. HippoRAG (纯语义):")
if hipporag_eval:
    print(f"     Recall@1: {hipporag_eval.get('Recall@1', 0.0)*100:.1f}%")
    print(f"     Recall@5: {hipporag_eval.get('Recall@5', 0.0)*100:.1f}%")
print("  2. HyperAmy (Emos模型 + GPU):")
if hyperamy_eval:
    print(f"     Recall@1: {hyperamy_eval.get('Recall@1', 0.0)*100:.1f}%")
    print(f"     Recall@5: {hyperamy_eval.get('Recall@5', 0.0)*100:.1f}%")
print("=" * 80)

# 保存结果
results_file = output_dir / "results.json"  # 使用标准文件名以便对比
results_data = {
    'timestamp': datetime.now().isoformat(),
    'device': device,
    'model': 'EmosWrapper',
    'hipporag_eval': hipporag_eval,
    'hyperamy_eval': hyperamy_eval,
    'stored_points': stored_points
}
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)
logger.info(f"✅ 结果已保存到: {results_file}")

print(f"\n✅ 实验完成！结果已保存到: {results_file}")
