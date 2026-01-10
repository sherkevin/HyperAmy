#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fusion策略网格搜索实验

测试不同的融合策略、归一化方法和sentiment_weight组合，找到最优配置。
总配置数: 4种策略 × 4种归一化 × 5个权重 = 80个配置

优化措施：
- 复用已有索引（HippoRAG语义索引 + Fusion情绪向量）
- 并发执行多个配置测试
- 启用情绪向量提取缓存
- 使用统一评估接口
- 支持断点续传
"""

import sys
import os
import json
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.config import API_KEY, BASE_URL, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
from sentiment.hipporag_enhanced import HippoRAGEnhanced
from sentiment.fusion_strategies import FusionStrategy, NormalizationStrategy
from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from evaluation.unified_evaluator import UnifiedEvaluator
from evaluation.report_generator import ReportGenerator

# 设置环境变量
os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['API_KEY'] = API_KEY

# 配置日志
log_file = project_root / 'fusion_strategy_grid_search.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型配置
LLM_MODEL_NAME = DEFAULT_MODEL
LLM_BASE_URL = BASE_URL
EMBEDDING_MODEL_NAME = f"VLLM/{DEFAULT_EMBEDDING_MODEL}"
EMBEDDING_BASE_URL = API_URL_EMBEDDINGS

# 实验配置
FUSION_STRATEGIES = [
    FusionStrategy.LINEAR,
    FusionStrategy.HARMONIC,
    FusionStrategy.GEOMETRIC,
    FusionStrategy.RANK_FUSION
]

NORMALIZATION_STRATEGIES = [
    NormalizationStrategy.MIN_MAX,
    NormalizationStrategy.Z_SCORE,
    NormalizationStrategy.L2,
    NormalizationStrategy.NONE
]

SENTIMENT_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]

# 输出配置
OUTPUT_DIR = project_root / "outputs" / "fusion_strategy_grid_search"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 进度文件（支持断点续传）
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 并发配置
MAX_WORKERS = 15  # 同时测试15个配置（提高并发度）
CONFIG_BATCH_SIZE = 30  # 每批处理30个配置（减少批次切换开销）


def load_progress() -> Dict[str, Any]:
    """加载实验进度"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
    return {
        'completed_configs': [],
        'failed_configs': [],
        'start_time': datetime.now().isoformat()
    }


def save_progress(progress: Dict[str, Any]):
    """保存实验进度"""
    progress['last_update'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def get_config_key(strategy: FusionStrategy, norm: NormalizationStrategy, weight: float) -> str:
    """生成配置的唯一键"""
    return f"{strategy.value}_{norm.value}_{weight:.1f}"


def load_dataset():
    """加载数据集"""
    logger.info("Loading dataset...")
    
    chunks_file = project_root / "data" / "training" / "monte_cristo_train_full.jsonl"
    qa_file = project_root / "data" / "public_benchmark" / "monte_cristo_qa_full.json"
    
    if not chunks_file.exists():
        raise FileNotFoundError(f"Training data not found: {chunks_file}")
    if not qa_file.exists():
        raise FileNotFoundError(f"QA data not found: {qa_file}")
    
    # 加载chunks
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # 准备文档列表
    docs = []
    chunk_id_to_doc = {}
    for chunk_idx, chunk in enumerate(chunks):
        text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
        chunk_id = chunk.get('chunk_id') or chunk.get('id') or f'chunk_{chunk_idx}'
        
        if isinstance(text, str) and len(text.strip()) > 20:
            docs.append(text.strip())
            chunk_id_to_doc[chunk_id] = text.strip()
    
    logger.info(f"Prepared {len(docs)} documents for indexing")
    
    # 加载QA数据
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    queries = [qa['question'] for qa in qa_pairs]
    
    # 准备gold_docs
    gold_docs = []
    for qa in qa_pairs:
        chunk_id = qa.get('chunk_id')
        gold_text = chunk_id_to_doc.get(chunk_id) if chunk_id else None
        if gold_text:
            gold_docs.append([gold_text])
        else:
            gold_docs.append([])
    
    return docs, queries, gold_docs, qa_pairs


def ensure_base_indexes(docs: List[str], base_output_dir: Path):
    """确保基础索引存在（HippoRAG和Fusion的基础索引）"""
    logger.info("Checking base indexes...")
    
    # 检查HippoRAG索引
    hipporag_save_dir = base_output_dir / "hipporag"
    hipporag_config = BaseConfig(
        save_dir=str(hipporag_save_dir),
        llm_base_url=LLM_BASE_URL,
        llm_name=LLM_MODEL_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        embedding_base_url=EMBEDDING_BASE_URL,
        force_index_from_scratch=False,
        retrieval_top_k=5,
    )
    
    hipporag_working_dir = Path(hipporag_config.save_dir) / f"{LLM_MODEL_NAME.replace('/', '_')}_{EMBEDDING_MODEL_NAME.replace('/', '_')}"
    
    # 检查HippoRAG索引是否存在（检查关键目录和文件）
    has_hipporag_index = (
        hipporag_working_dir.exists() and
        (hipporag_working_dir / "chunk_embeddings").exists() and
        len(list((hipporag_working_dir / "chunk_embeddings").glob("*"))) > 0
    )
    
    if not has_hipporag_index:
        logger.info("Building HippoRAG base index...")
        hipporag = HippoRAG(global_config=hipporag_config)
        hipporag.index(docs=docs)
        logger.info("✅ HippoRAG base index ready")
    else:
        logger.info("✅ HippoRAG base index exists, reusing")
    
    # 检查Fusion基础索引（情绪向量）
    # 使用已有的fusion目录（如果存在）
    fusion_save_dir = base_output_dir / "fusion"
    if not (fusion_save_dir / f"{LLM_MODEL_NAME.replace('/', '_')}_{EMBEDDING_MODEL_NAME.replace('/', '_')}").exists():
        fusion_save_dir = base_output_dir / "fusion_base"
    
    fusion_config = BaseConfig(
        save_dir=str(fusion_save_dir),
        llm_base_url=LLM_BASE_URL,
        llm_name=LLM_MODEL_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        embedding_base_url=EMBEDDING_BASE_URL,
        force_index_from_scratch=False,
        retrieval_top_k=5,
    )
    
    fusion_working_dir = Path(fusion_config.save_dir) / f"{LLM_MODEL_NAME.replace('/', '_')}_{EMBEDDING_MODEL_NAME.replace('/', '_')}"
    emotion_store_dir = fusion_working_dir / "chunk_emotions"
    emotion_file = emotion_store_dir / "emotion_emotion.parquet"
    
    has_emotion_index = emotion_file.exists()
    
    if not has_emotion_index:
        logger.info("Building Fusion base index (emotion vectors)...")
        fusion_base = HippoRAGEnhanced(
            global_config=fusion_config,
            enable_sentiment=True,
            sentiment_weight=0.5,  # 使用默认权重构建基础索引
            sentiment_model_name=LLM_MODEL_NAME,
            max_workers=15,  # 使用更多worker加速
            fusion_strategy=FusionStrategy.LINEAR,  # 使用默认策略
            normalization_strategy=NormalizationStrategy.MIN_MAX
        )
        fusion_base.index(docs=docs)
        logger.info("✅ Fusion base index (emotion vectors) ready")
    else:
        logger.info("✅ Fusion base index (emotion vectors) exists, reusing")
    
    return hipporag_save_dir, fusion_save_dir


def test_single_config(
    config_key: str,
    strategy: FusionStrategy,
    normalization: NormalizationStrategy,
    sentiment_weight: float,
    docs: List[str],
    queries: List[str],
    gold_docs: List[List[str]],
    base_fusion_dir: Path,
    base_output_dir: Path
) -> Dict[str, Any]:
    """
    测试单个配置
    
    注意：我们复用基础索引，只需要运行检索阶段，应用不同的融合策略。
    """
    result_file = RESULTS_DIR / f"result_{config_key}.json"
    
    # 如果结果已存在，直接加载
    if result_file.exists():
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            logger.info(f"Config {config_key} already completed, skipping")
            return result
        except Exception as e:
            logger.warning(f"Failed to load existing result for {config_key}: {e}")
    
    try:
        start_time = time.time()
        
        # 为每个配置使用独立的save_dir，避免并发冲突
        # 但通过force_index_from_scratch=False和符号链接来复用基础索引
        config_save_dir = OUTPUT_DIR / "configs" / config_key
        config_save_dir.mkdir(parents=True, exist_ok=True)
        
        config = BaseConfig(
            save_dir=str(config_save_dir),
            llm_base_url=LLM_BASE_URL,
            llm_name=LLM_MODEL_NAME,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            embedding_base_url=EMBEDDING_BASE_URL,
            force_index_from_scratch=False,  # 复用索引
            retrieval_top_k=5,
        )
        
        # 计算working_dir路径（在创建实例之前）
        working_dir_name = f"{LLM_MODEL_NAME.replace('/', '_')}_{EMBEDDING_MODEL_NAME.replace('/', '_')}"
        config_working_dir = config_save_dir / working_dir_name
        
        # 复用基础索引（HippoRAG的语义索引）
        base_hipporag_dir = base_output_dir / "hipporag" / working_dir_name
        if base_hipporag_dir.exists() and not config_working_dir.exists():
            # 创建符号链接复用语义索引
            try:
                import os
                os.symlink(str(base_hipporag_dir), str(config_working_dir))
                logger.debug(f"Created symlink to reuse HippoRAG index for {config_key}")
            except (OSError, NotImplementedError):
                # 如果不支持symlink，创建软链接目录结构
                logger.debug(f"Symlink not supported, will use separate index (will be fast due to force_index_from_scratch=False)")
        
        # 准备复用情绪向量存储
        base_emotion_store_dir = base_fusion_dir / working_dir_name / "chunk_emotions"
        base_emotion_file = base_emotion_store_dir / "emotion_emotion.parquet"
        config_emotion_store_dir = config_working_dir / "chunk_emotions"
        
        # 如果基础情绪向量存在，创建符号链接或复制以复用
        if base_emotion_file.exists():
            if config_working_dir.exists() and not config_emotion_store_dir.exists():
                config_emotion_store_dir.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import os
                    os.symlink(str(base_emotion_store_dir), str(config_emotion_store_dir))
                    logger.debug(f"Created symlink to reuse emotion vectors for {config_key}")
                except (OSError, NotImplementedError):
                    # 如果不支持symlink，复制（只复制一次）
                    import shutil
                    try:
                        shutil.copytree(base_emotion_store_dir, config_emotion_store_dir)
                        logger.debug(f"Copied emotion vectors to {config_key}")
                    except Exception as copy_e:
                        logger.warning(f"Failed to copy emotion vectors: {copy_e}")
        
        # 初始化Fusion（使用指定配置）
        fusion = HippoRAGEnhanced(
            global_config=config,
            enable_sentiment=True,
            sentiment_weight=sentiment_weight,
            sentiment_model_name=LLM_MODEL_NAME,
            max_workers=10,
            fusion_strategy=strategy,
            normalization_strategy=normalization
        )
        
        # 执行检索（会复用已有的语义索引和情绪向量）
        logger.info(f"Testing config {config_key}: strategy={strategy.value}, norm={normalization.value}, weight={sentiment_weight}")
        
        # 关键优化：检查是否可以直接跳过index()
        # 如果working_dir已存在且包含必要的索引文件，可以尝试直接检索
        # 但为了安全起见，仍然调用index()，因为它会快速跳过已有部分
        working_dir_path = Path(fusion.working_dir)
        has_semantic_index = (
            working_dir_path.exists() and
            (working_dir_path / "chunk_embeddings").exists() and
            len(list((working_dir_path / "chunk_embeddings").glob("*"))) > 0
        )
        has_emotion_index = config_emotion_store_dir.exists() and base_emotion_file.exists()
        
        if has_semantic_index and has_emotion_index:
            logger.debug(f"Config {config_key}: Both semantic and emotion indices exist, index() will skip")
        
        # 运行index() - 会自动检测已有情绪向量和语义索引并跳过（如果已存在）
        # 由于我们设置了force_index_from_scratch=False，且基础索引已存在，index()会很快完成
        # 情绪向量会通过emotion_store.contains()检查自动跳过
        fusion.index(docs=docs)
        
        # 执行检索（这是每个配置的主要工作）
        retrieval_results = fusion.retrieve(
            queries=queries,
            num_to_retrieve=5,
            gold_docs=gold_docs
        )
        
        # 处理返回结果
        if isinstance(retrieval_results, tuple):
            fusion_results, fusion_eval = retrieval_results
        else:
            fusion_results = retrieval_results
            fusion_eval = None
        
        # 提取检索结果文本
        retrieved_texts_list = [[r.docs] if hasattr(r, 'docs') else [] for r in fusion_results]
        # 修正：retrieve返回的是QuerySolution列表
        retrieved_texts_list = [r.docs if hasattr(r, 'docs') else [] for r in fusion_results]
        gold_texts = [gd[0] if gd else "" for gd in gold_docs]
        
        # 使用统一评估接口
        evaluator = UnifiedEvaluator(k_list=[1, 2, 5, 10])
        eval_result = evaluator.evaluate_exact_match(
            method_name=f"Fusion_{config_key}",
            gold_texts=gold_texts,
            retrieved_texts_list=retrieved_texts_list,
            k_list=[1, 2, 5, 10]
        )
        
        elapsed_time = time.time() - start_time
        
        result = {
            'config_key': config_key,
            'strategy': strategy.value,
            'normalization': normalization.value,
            'sentiment_weight': sentiment_weight,
            'metrics': {
                'recall_at_k': eval_result.recall_at_k,
                'mrr': eval_result.mrr,
                'map': eval_result.map,
                'ndcg_at_k': eval_result.ndcg_at_k,
                'precision_at_k': eval_result.precision_at_k,
                'f1_at_k': eval_result.f1_at_k,
            },
            'hipporag_eval': fusion_eval,  # 保留原始的HippoRAG评估结果
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存结果
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Config {config_key} completed in {elapsed_time:.2f}s - Recall@10={eval_result.recall_at_k.get(10, 0.0):.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Config {config_key} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_key': config_key,
            'strategy': strategy.value,
            'normalization': normalization.value,
            'sentiment_weight': sentiment_weight,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def rerun_failed_configs():
    """重新运行失败的配置"""
    logger.info("=" * 80)
    logger.info("Re-running Failed Configurations")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # 加载进度
    progress = load_progress()
    failed_keys = set(progress.get('failed_configs', []))
    
    if not failed_keys:
        logger.info("No failed configurations to rerun!")
        return
    
    logger.info(f"Found {len(failed_keys)} failed configurations")
    
    # 加载数据集
    docs, queries, gold_docs, qa_pairs = load_dataset()
    
    # 确保基础索引存在
    base_output_dir = project_root / "outputs" / "three_methods_comparison_monte_cristo"
    hipporag_dir, fusion_base_dir = ensure_base_indexes(docs, base_output_dir)
    
    # 解析失败的配置键
    failed_configs = []
    for config_key in failed_keys:
        # 解析配置键格式: "strategy_normalization_weight"
        parts = config_key.split('_')
        if len(parts) >= 3:
            try:
                strategy_str = parts[0]
                norm_str = '_'.join(parts[1:-1])  # 处理可能包含下划线的归一化策略
                weight = float(parts[-1])
                
                # 转换为枚举
                strategy = FusionStrategy(strategy_str)
                normalization = NormalizationStrategy(norm_str)
                
                failed_configs.append((strategy, normalization, weight))
            except Exception as e:
                logger.warning(f"Failed to parse config key {config_key}: {e}")
    
    logger.info(f"Parsed {len(failed_configs)} configurations to rerun")
    
    # 清除失败标记（重新测试）
    progress['failed_configs'] = []
    save_progress(progress)
    
    # 删除失败配置的结果文件（强制重新运行）
    for config_key in failed_keys:
        result_file = RESULTS_DIR / f"result_{config_key}.json"
        if result_file.exists():
            try:
                # 检查是否包含错误（如果包含，删除以重新运行）
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    if 'error' in result_data:
                        result_file.unlink()
                        logger.info(f"Deleted failed result file: {result_file}")
            except Exception as e:
                logger.warning(f"Failed to check/delete result file {result_file}: {e}")
    
    # 运行失败的配置
    completed_keys = set(progress.get('completed_configs', []))
    new_failed_keys = set()
    
    # 分批处理
    num_batches = (len(failed_configs) + CONFIG_BATCH_SIZE - 1) // CONFIG_BATCH_SIZE
    
    for batch_idx in range(num_batches):
        batch_configs = failed_configs[batch_idx * CONFIG_BATCH_SIZE:(batch_idx + 1) * CONFIG_BATCH_SIZE]
        logger.info(f"\n{'='*80}")
        logger.info(f"Rerun Batch {batch_idx + 1}/{num_batches}: Testing {len(batch_configs)} configurations")
        logger.info(f"{'='*80}")
        
        # 并发执行当前批次的配置
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_config = {
                executor.submit(
                    test_single_config,
                    get_config_key(strategy, norm, weight),
                    strategy,
                    norm,
                    weight,
                    docs,
                    queries,
                    gold_docs,
                    fusion_base_dir,
                    base_output_dir
                ): (strategy, norm, weight)
                for strategy, norm, weight in batch_configs
            }
            
            batch_results = []
            for future in tqdm(as_completed(future_to_config), total=len(batch_configs), desc=f"Rerun Batch {batch_idx+1}"):
                strategy, norm, weight = future_to_config[future]
                config_key = get_config_key(strategy, norm, weight)
                
                try:
                    result = future.result()
                    if 'error' not in result:
                        completed_keys.add(config_key)
                        batch_results.append(result)
                        progress['completed_configs'] = list(completed_keys)
                        if config_key in new_failed_keys:
                            new_failed_keys.remove(config_key)
                    else:
                        new_failed_keys.add(config_key)
                        progress['failed_configs'] = list(new_failed_keys)
                        batch_results.append(result)
                    
                    # 每完成一个配置就保存进度
                    save_progress(progress)
                    
                except Exception as e:
                    logger.error(f"Unexpected error for config {config_key}: {e}")
                    new_failed_keys.add(config_key)
                    progress['failed_configs'] = list(new_failed_keys)
                    save_progress(progress)
        
        logger.info(f"Rerun Batch {batch_idx + 1} completed: {len([r for r in batch_results if 'error' not in r])} successful, "
                   f"{len([r for r in batch_results if 'error' in r])} failed")
    
    # 生成汇总报告
    logger.info("\n" + "="*80)
    logger.info("Generating summary report...")
    generate_summary_report()
    logger.info("="*80)
    logger.info(f"Rerun completed! Successful: {len(completed_keys)}, Still failed: {len(new_failed_keys)}")


def run_grid_search():
    """运行网格搜索实验"""
    logger.info("=" * 80)
    logger.info("Fusion Strategy Grid Search Experiment")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total configurations: {len(FUSION_STRATEGIES) * len(NORMALIZATION_STRATEGIES) * len(SENTIMENT_WEIGHTS)}")
    logger.info("=" * 80)
    
    # 加载数据集
    docs, queries, gold_docs, qa_pairs = load_dataset()
    
    # 确保基础索引存在
    base_output_dir = project_root / "outputs" / "three_methods_comparison_monte_cristo"
    hipporag_dir, fusion_base_dir = ensure_base_indexes(docs, base_output_dir)
    
    # 加载进度
    progress = load_progress()
    completed_keys = set(progress.get('completed_configs', []))
    failed_keys = set(progress.get('failed_configs', []))
    
    logger.info(f"Progress: {len(completed_keys)} completed, {len(failed_keys)} failed")
    
    # 生成所有配置
    all_configs = list(itertools.product(
        FUSION_STRATEGIES,
        NORMALIZATION_STRATEGIES,
        SENTIMENT_WEIGHTS
    ))
    
    total_configs = len(all_configs)
    logger.info(f"Total configurations to test: {total_configs}")
    
    # 过滤已完成的配置
    remaining_configs = [
        (s, n, w) for s, n, w in all_configs
        if get_config_key(s, n, w) not in completed_keys
    ]
    
    logger.info(f"Remaining configurations: {len(remaining_configs)}")
    
    if not remaining_configs:
        logger.info("All configurations already completed!")
        return
    
    # 分批处理配置
    num_batches = (len(remaining_configs) + CONFIG_BATCH_SIZE - 1) // CONFIG_BATCH_SIZE
    
    for batch_idx in range(num_batches):
        batch_configs = remaining_configs[batch_idx * CONFIG_BATCH_SIZE:(batch_idx + 1) * CONFIG_BATCH_SIZE]
        logger.info(f"\n{'='*80}")
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: Testing {len(batch_configs)} configurations")
        logger.info(f"{'='*80}")
        
        # 并发执行当前批次的配置
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_config = {
                executor.submit(
                    test_single_config,
                    get_config_key(strategy, norm, weight),
                    strategy,
                    norm,
                    weight,
                    docs,
                    queries,
                    gold_docs,
                    fusion_base_dir,
                    base_output_dir
                ): (strategy, norm, weight)
                for strategy, norm, weight in batch_configs
            }
            
            batch_results = []
            for future in tqdm(as_completed(future_to_config), total=len(batch_configs), desc=f"Batch {batch_idx+1}"):
                strategy, norm, weight = future_to_config[future]
                config_key = get_config_key(strategy, norm, weight)
                
                try:
                    result = future.result()
                    if 'error' not in result:
                        completed_keys.add(config_key)
                        batch_results.append(result)
                        progress['completed_configs'] = list(completed_keys)
                    else:
                        failed_keys.add(config_key)
                        progress['failed_configs'] = list(failed_keys)
                        batch_results.append(result)
                    
                    # 每完成一个配置就保存进度
                    save_progress(progress)
                    
                except Exception as e:
                    logger.error(f"Unexpected error for config {config_key}: {e}")
                    failed_keys.add(config_key)
                    progress['failed_configs'] = list(failed_keys)
                    save_progress(progress)
        
        logger.info(f"Batch {batch_idx + 1} completed: {len([r for r in batch_results if 'error' not in r])} successful, "
                   f"{len([r for r in batch_results if 'error' in r])} failed")
    
    # 生成汇总报告
    logger.info("\n" + "="*80)
    logger.info("Generating summary report...")
    generate_summary_report()
    logger.info("="*80)
    logger.info(f"Grid search completed! Total: {len(completed_keys)} successful, {len(failed_keys)} failed")


def generate_summary_report():
    """生成汇总报告"""
    # 收集所有结果
    all_results = []
    for result_file in RESULTS_DIR.glob("result_*.json"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                if 'error' not in result:
                    all_results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")
    
    if not all_results:
        logger.warning("No valid results found for report generation")
        return
    
    # 找出最佳配置
    best_recall_10 = max(all_results, key=lambda r: r.get('metrics', {}).get('recall_at_k', {}).get(10, 0.0))
    best_mrr = max(all_results, key=lambda r: r.get('metrics', {}).get('mrr', 0.0) or 0.0)
    best_map = max(all_results, key=lambda r: r.get('metrics', {}).get('map', 0.0) or 0.0)
    
    summary = {
        'total_configs': len(all_results),
        'experiment_date': datetime.now().isoformat(),
        'best_configs': {
            'recall@10': {
                'config_key': best_recall_10['config_key'],
                'strategy': best_recall_10['strategy'],
                'normalization': best_recall_10['normalization'],
                'sentiment_weight': best_recall_10['sentiment_weight'],
                'recall@10': best_recall_10['metrics']['recall_at_k'].get(10, 0.0),
                'all_metrics': best_recall_10['metrics']
            },
            'mrr': {
                'config_key': best_mrr['config_key'],
                'strategy': best_mrr['strategy'],
                'normalization': best_mrr['normalization'],
                'sentiment_weight': best_mrr['sentiment_weight'],
                'mrr': best_mrr['metrics'].get('mrr', 0.0),
                'all_metrics': best_mrr['metrics']
            },
            'map': {
                'config_key': best_map['config_key'],
                'strategy': best_map['strategy'],
                'normalization': best_map['normalization'],
                'sentiment_weight': best_map['sentiment_weight'],
                'map': best_map['metrics'].get('map', 0.0),
                'all_metrics': best_map['metrics']
            }
        },
        'all_results': all_results
    }
    
    # 保存汇总JSON
    summary_file = OUTPUT_DIR / "grid_search_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary saved to {summary_file}")
    
    # 生成HTML报告（使用统一评估接口的结果格式）
    try:
        from evaluation.unified_evaluator import UnifiedEvaluationResult
        
        # 转换为UnifiedEvaluationResult格式
        eval_results = []
        for result in all_results[:10]:  # 只取前10个生成报告，避免太大
            if 'metrics' in result:
                eval_result = UnifiedEvaluationResult(
                    method_name=f"Fusion_{result['config_key']}",
                    evaluation_mode='exact_match',
                    recall_at_k=result['metrics'].get('recall_at_k', {}),
                    mrr=result['metrics'].get('mrr'),
                    map=result['metrics'].get('map'),
                    ndcg_at_k=result['metrics'].get('ndcg_at_k', {}),
                    precision_at_k=result['metrics'].get('precision_at_k', {}),
                    f1_at_k=result['metrics'].get('f1_at_k', {})
                )
                eval_results.append(eval_result)
        
        if eval_results:
            generator = ReportGenerator(output_dir=str(OUTPUT_DIR))
            html_file = generator.generate_html_report(
                results=eval_results,
                title="Fusion Strategy Grid Search Results"
            )
            logger.info(f"HTML report saved to {html_file}")
    
    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {e}")
    
    # 打印最佳配置摘要
    logger.info("\n" + "="*80)
    logger.info("Best Configurations:")
    logger.info("="*80)
    logger.info(f"Best Recall@10: {best_recall_10['config_key']}")
    logger.info(f"  Strategy: {best_recall_10['strategy']}, Norm: {best_recall_10['normalization']}, Weight: {best_recall_10['sentiment_weight']}")
    logger.info(f"  Recall@10: {best_recall_10['metrics']['recall_at_k'].get(10, 0.0):.4f}")
    logger.info(f"  MRR: {best_recall_10['metrics'].get('mrr', 0.0):.4f}")
    logger.info("")
    logger.info(f"Best MRR: {best_mrr['config_key']}")
    logger.info(f"  Strategy: {best_mrr['strategy']}, Norm: {best_mrr['normalization']}, Weight: {best_mrr['sentiment_weight']}")
    logger.info(f"  MRR: {best_mrr['metrics'].get('mrr', 0.0):.4f}")
    logger.info(f"  Recall@10: {best_mrr['metrics']['recall_at_k'].get(10, 0.0):.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fusion策略网格搜索实验")
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="重新运行失败的配置"
    )
    
    args = parser.parse_args()
    
    try:
        if args.rerun_failed:
            rerun_failed_configs()
            logger.info("="*80)
            logger.info("Rerun failed configurations completed!")
            logger.info(f"Results directory: {OUTPUT_DIR}")
            logger.info(f"Summary file: {OUTPUT_DIR / 'grid_search_summary.json'}")
            logger.info("="*80)
        else:
            run_grid_search()
            logger.info("="*80)
            logger.info("Grid search experiment completed successfully!")
            logger.info(f"Results directory: {OUTPUT_DIR}")
            logger.info(f"Summary file: {OUTPUT_DIR / 'grid_search_summary.json'}")
            logger.info("="*80)
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        save_progress(load_progress())
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

