#!/usr/bin/env python3
"""
H-Mem System V3 对比测试：HippoRAG vs HyperAmy V3 vs GraphFusion

测试三种检索方案：
1. HippoRAG: 纯图检索方法（基准）
2. HyperAmy V3: 基于双曲空间和引力的记忆系统（新）
3. GraphFusion: 融合 HippoRAG 图谱和 HyperAmy 情感的方法

评估维度：
- 检索性能（时间）
- 检索准确性（排名）
- V3 物理特性（质量衰减、温度调制）
"""
import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np

# 设置环境变量
from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
os.environ["OPENAI_API_KEY"] = API_KEY

import logging
log_file = Path("./log/test_v3_comparison.log")
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from workflow.graph_fusion_retrieval import GraphFusionRetriever
from workflow.amygdala import Amygdala
from workflow.hipporag_wrapper import HippoRAGWrapper

# V3 组件
from poincare.storage import HyperAmyStorage
from poincare.retrieval import InMemoryRetrieval, RetrievalConfig, create_candidate
from poincare.physics import PhysicsEngine
from particle.properties import PropertyCalculator

logger.info("=" * 120)
logger.info("H-Mem System V3 对比测试")
logger.info("=" * 120)
logger.info("对比方案: HippoRAG | HyperAmy V3 | GraphFusion")
logger.info("=" * 120)

# ========== 测试数据（Monte Cristo场景）==========
chunks = [
    '"I have an excellent appetite," said Albert. "I hope, my dear Count, you have the same." '
    '"I?" said Monte Cristo. "I never eat, or rather, I eat so little that it is not worth '
    'talking about. I have my own peculiar habits."',

    'The Count took from his pocket a small case made of hollowed emerald, took out a small '
    'greenish pill, and swallowed it. "This is my food," he said to the guests. "With this, '
    'I feel neither hunger nor fatigue. It is a secret I learned in the East."',

    '"Will you not take anything?" asked Mercedes. "A peach? Some grapes?" '
    '"I thank you, Madame," replied Monte Cristo with a bow, "but I never eat between meals. '
    'It is a rule I have imposed upon myself to maintain my health."',

    '"In the countries of the East, where I have lived," said Monte Cristo to Franz, '
    '"people who eat and drink together are bound by a sacred tie. They become brothers. '
    'Therefore, I never eat or drink in the house of a man whom I wish to kill. '
    'If I shared their bread, I would be forbidden by honor to take my revenge."',

    'Mercedes looked at him with terror in her eyes. Her hand trembled as she held the plate. '
    '"You refuse?" she whispered, her voice full of tears. "Is it because you are our enemy? '
    'To refuse to break bread... means you bring death to this house." '
    'She realized then that the man standing before her was not just a visitor, but an avenger '
    'who remembered the past.'
]

chunk_types = {
    0: "早餐场景",
    1: "药丸场景",
    2: "拒绝葡萄干",
    3: "东方哲学（正确答案）",
    4: "情感对峙"
}

query = "Why did the Count strictly refuse the muscatel grapes and any refreshment offered by Madame de Morcerf (Mercedes) during his visit to her house?"

# 期望答案排名
expected_ranking = [3, 4, 2, 1, 0]  # 东方哲学第1，情感对峙第2...

# ========== Step 1: 清理旧数据 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 1】清理旧数据")
logger.info("=" * 120)

test_databases = [
    "./test_v3_amygdala_db",
    "./test_v3_hipporag_db",
    "./hyperamy_db_v3_particles",
    "./hyperamy_db_v3_fusion"
]

for db_dir in test_databases:
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        logger.info(f"✓ 删除 {db_dir}")

logger.info("✓ 旧数据清理完成")

# ========== Step 2: 创建 Amygdala 数据库（用于 V3 和 Fusion）==========
logger.info("\n" + "=" * 120)
logger.info("【Step 2】创建 Amygdala 数据库")
logger.info("=" * 120)

amygdala = Amygdala(
    save_dir="./test_v3_amygdala_db",
    particle_collection_name="v3_test_particles",
    conversation_namespace="v3_test",
    auto_link_particles=False
)
logger.info("✓ Amygdala 初始化完成")

logger.info(f"\n添加 {len(chunks)} 个对话...")
for i, chunk in enumerate(chunks):
    result = amygdala.add(chunk)
    logger.info(f"  Chunk {i+1}: 生成 {result['particle_count']} 个粒子")

logger.info(f"✓ Amygdala 数据库创建完成（{len(chunks)} 个对话）")

# ========== Step 3: 创建 HippoRAG 数据库 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 3】创建 HippoRAG 数据库")
logger.info("=" * 120)

hipporag = HippoRAGWrapper(
    save_dir="./test_v3_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    llm_base_url=BASE_URL,
    embedding_model_name=f"VLLM/{DEFAULT_EMBEDDING_MODEL}",
    embedding_base_url=API_URL_EMBEDDINGS
)
logger.info("✓ HippoRAG 初始化完成")

logger.info(f"\n添加 {len(chunks)} 个对话...")
hipporag.add(chunks)
logger.info(f"✓ HippoRAG 数据库创建完成（{len(chunks)} 个对话）")

# ========== Step 4: 创建 GraphFusion ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 4】创建 GraphFusion 检索器")
logger.info("=" * 120)

fusion = GraphFusionRetriever(
    amygdala_save_dir="./test_v3_amygdala_db",
    hipporag_save_dir="./test_v3_hipporag_db",
    llm_model_name="DeepSeek-V3.2",
    auto_link_particles=False
)
logger.info("✓ GraphFusion 初始化完成")

# ========== Step 5: 初始化 V3 组件 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 5】初始化 HyperAmy V3 组件")
logger.info("=" * 120)

# V3 物理引擎和属性计算器
physics = PhysicsEngine(curvature=1.0, gamma=0.01)  # 同样调整 gamma
calc = PropertyCalculator()

# V3 检索器配置
# gamma 调整为 0.01 以适配秒级时间尺度（原始设计为天级）
# 原始: gamma=1.0, 时间单位为天 (86400秒)
# 现在: gamma=0.01, 时间单位为秒，相当于每天衰减 0.01*86400 = 864
config_v3 = RetrievalConfig(
    semantic_threshold=0.3,   # 较低的阈值，允许更多候选
    retrieval_beta=1.0,        # 温度调制系数
    curvature=1.0,
    gamma=0.01,               # 调整为适配秒级时间
    forgetting_threshold=1e-3  # 遗忘阈值
)
v3_retrieval = InMemoryRetrieval(config=config_v3)

logger.info("✓ V3 组件初始化完成")

# 从 Amygdala 数据库加载粒子到 V3 检索器
logger.info("\n加载粒子到 V3 检索器...")
all_candidates = amygdala.particle_storage.collection.get(include=["embeddings", "metadatas"])
ids = all_candidates.get("ids", [])
embeddings = all_candidates.get("embeddings", [])
metadatas = all_candidates.get("metadatas", [])

loaded_count = 0
particle_born_times = []  # 记录所有粒子的出生时间

for i, pid in enumerate(ids):
    if i < len(embeddings) and embeddings[i] is not None:
        direction = np.array(embeddings[i], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

    if i < len(metadatas) and metadatas[i] is not None:
        meta = metadatas[i]
        mass = float(meta.get("weight", 1.0))
        temperature = float(meta.get("T", 1.0))
        speed = float(meta.get("v", 0.5))
        born = float(meta.get("born", time.time()))
        particle_born_times.append(born)

        # 根据 speed 计算初始半径（V3 物理模型）
        initial_radius = calc.compute_initial_radius(mass=mass)

        candidate = create_candidate(
            particle_id=pid,
            direction=direction,
            mass=mass,
            temperature=temperature,
            initial_radius=initial_radius,
            created_at=born,
            conversation_id=meta.get("conversation_id", ""),
            entity=meta.get("entity", "")
        )
        v3_retrieval.add_particle(candidate)
        loaded_count += 1

logger.info(f"✓ 加载了 {loaded_count} 个粒子到 V3 检索器")

# 计算粒子的时间范围（用于设置查询时间）
if particle_born_times:
    min_born = min(particle_born_times)
    max_born = max(particle_born_times)
    logger.info(f"  粒子时间范围: {min_born:.2f} ~ {max_born:.2f} (跨度: {max_born - min_born:.2f} 秒)")

# ========== Step 6: 处理查询 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 6】处理查询并生成查询粒子")
logger.info("=" * 120)
logger.info(f"Query: {query}")

# 处理查询生成粒子（计时，用于公平比较）
start_time = time.time()
query_particles = amygdala.particle.process(
    text=query,
    text_id=f"v3_query_{int(time.time())}"
)
v3_query_time = time.time() - start_time

logger.info(f"✓ 查询生成 {len(query_particles)} 个粒子（耗时: {v3_query_time:.2f}s）")

if query_particles:
    qp = query_particles[0]
    logger.info(f"  查询粒子: entity={qp.entity}, speed={qp.speed}, temperature={qp.temperature}, weight={qp.weight}")

# ========== Step 7: 测试三种检索方案 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 7】测试三种检索方案")
logger.info("=" * 120)

# ===== 测试 1: HippoRAG =====
logger.info("\n" + "-" * 120)
logger.info("【测试 1】HippoRAG 检索（基准）")
logger.info("-" * 120)

start_time = time.time()
hipporag_results_raw = hipporag.retrieve(query=query, top_k=5)
hipporag_time = time.time() - start_time

hipporag_results = []
for rank, result in enumerate(hipporag_results_raw):
    hipporag_results.append({
        'rank': rank + 1,
        'text': result['text'],
        'score': result['score']
    })

logger.info(f"✓ HippoRAG 检索完成（耗时: {hipporag_time:.2f}s）")
for result in hipporag_results:
    chunk_type = chunk_types.get(chunks.index(result['text']), "Unknown")
    logger.info(f"  Rank {result['rank']}: {chunk_type} (分数: {result['score']:.4f})")

# ===== 测试 2: HyperAmy V3 =====
logger.info("\n" + "-" * 120)
logger.info("【测试 2】HyperAmy V3 检索（双曲空间 + 引力时间膨胀）")
logger.info("-" * 120)

if query_particles:
    # 记录所有查询粒子
    logger.info(f"  查询生成 {len(query_particles)} 个粒子:")
    for i, qp in enumerate(query_particles):
        logger.info(f"    粒子 {i+1}: entity={qp.entity}, speed={qp.speed:.4f}, temperature={qp.temperature:.4f}, weight={qp.weight:.4f}")

    # 使用所有查询粒子的平均向量（多粒子融合检索）
    logger.info("  使用多粒子融合检索策略...")
    query_directions = []
    query_masses = []
    query_temps = []

    for qp in query_particles:
        direction = qp.emotion_vector / np.linalg.norm(qp.emotion_vector)
        query_directions.append(direction)
        mass = calc.compute_mass(intensity=qp.speed, purity=1.0 / (qp.temperature + 0.1))
        temp = calc.compute_temperature(purity=1.0 / (qp.temperature + 0.1))
        query_masses.append(mass)
        query_temps.append(temp)

    # 平均方向
    query_direction = np.mean(query_directions, axis=0)
    query_direction = query_direction / np.linalg.norm(query_direction)
    query_mass = np.mean(query_masses)
    query_temperature = np.mean(query_temps)
    query_initial_radius = calc.compute_initial_radius(mass=query_mass)

    logger.info(f"  融合查询物理属性: mass={query_mass:.4f}, temperature={query_temperature:.4f}, R0={query_initial_radius:.4f}")

    # 使用与粒子创建时间相近的时间（避免遗忘）
    # 使用存储粒子的最新出生时间作为基准
    if particle_born_times:
        t_now_for_query = max_born + 10.0  # 查询时间比最晚的粒子晚 10 秒
    else:
        t_now_for_query = time.time()

    logger.info(f"  查询时间: t_now={t_now_for_query:.2f}, 粒子born范围: {min_born:.2f}~{max_born:.2f}")

    start_time = time.time()
    v3_results_raw = v3_retrieval.search(
        query_direction=query_direction,
        query_mass=query_mass,
        query_temperature=query_temperature,
        query_initial_radius=query_initial_radius,
        top_k=100,  # 获取更多候选结果用于聚合
        t_now=t_now_for_query
    )
    v3_time = time.time() - start_time

    # 转换结果格式
    # V3 返回的是粒子级别的结果，需要按对话聚合
    v3_results = []

    logger.info(f"  原始检索结果数量: {len(v3_results_raw)}")

    # 按对话聚合分数
    conversation_scores = {}  # {conv_id: [scores]}
    conversation_data = {}    # {conv_id: {text, max_score, avg_temp, etc.}}

    for result in v3_results_raw:
        conv_id = result.metadata.get("conversation_id", "")
        if not conv_id:
            continue

        if conv_id not in conversation_scores:
            conversation_scores[conv_id] = []
            conversation_data[conv_id] = {
                'max_score': result.score,
                'first_hyp_dist': result.hyperbolic_distance,
                'first_sem_sim': result.semantic_similarity,
                'first_temp': result.temperature,
                'first_memory': result.memory_strength,
                'particle_count': 0
            }

        conversation_scores[conv_id].append(result.score)
        conversation_data[conv_id]['max_score'] = max(conversation_data[conv_id]['max_score'], result.score)
        conversation_data[conv_id]['particle_count'] += 1

    # 获取文本并构建最终结果
    for conv_id, scores in conversation_scores.items():
        text = amygdala.conversation_store.get_text(conv_id)
        if text and text in chunks:
            chunk_idx = chunks.index(text)
            chunk_type = chunk_types.get(chunk_idx, "Unknown")

            # 使用平均分数作为排序依据
            avg_score = sum(scores) / len(scores)
            max_score = conversation_data[conv_id]['max_score']

            v3_results.append({
                'rank': 0,  # 稍后设置
                'text': text,
                'score': max_score,  # 使用最高分数
                'avg_score': avg_score,
                'hyperbolic_distance': conversation_data[conv_id]['first_hyp_dist'],
                'semantic_similarity': conversation_data[conv_id]['first_sem_sim'],
                'temperature': conversation_data[conv_id]['first_temp'],
                'memory_strength': conversation_data[conv_id]['first_memory'],
                'particle_count': conversation_data[conv_id]['particle_count']
            })

    # 按最高分数排序
    v3_results.sort(key=lambda x: x['score'], reverse=True)

    # 设置最终排名并限制为 top 5
    v3_results = v3_results[:5]
    for i, result in enumerate(v3_results):
        result['rank'] = i + 1

    logger.info(f"✓ HyperAmy V3 检索完成（检索耗时: {v3_time:.2f}s, 总耗时含查询处理: {v3_time + v3_query_time:.2f}s）")
    if v3_results:
        for result in v3_results:
            chunk_type = chunk_types.get(chunks.index(result['text']), "Unknown")
            logger.info(f"  Rank {result['rank']}: {chunk_type} (分数: {result['score']:.4f}, "
                       f"双曲距离: {result['hyperbolic_distance']:.4f}, "
                       f"语义相似度: {result['semantic_similarity']:.4f}, "
                       f"记忆强度: {result['memory_strength']:.4f}, "
                       f"粒子数: {result['particle_count']})")
    else:
        logger.info("  ⚠️ 未检索到结果")
else:
    v3_results = []
    v3_time = 0
    logger.info("  ⚠️ 查询粒子生成失败")

# ===== 测试 3: GraphFusion =====
logger.info("\n" + "-" * 120)
logger.info("【测试 3】GraphFusion 检索（图谱 + 情感融合）")
logger.info("-" * 120)

start_time = time.time()
fusion_results = fusion.retrieve(
    query=query,
    top_k=5,
    emotion_weight=0.3,
    semantic_weight=0.5,
    fact_weight=0.2
)
fusion_time = time.time() - start_time

logger.info(f"✓ GraphFusion 检索完成（耗时: {fusion_time:.2f}s）")
for result in fusion_results:
    chunk_type = chunk_types.get(chunks.index(result['text']), "Unknown")
    logger.info(f"  Rank {result['rank']}: {chunk_type} (PPR分数: {result['score']:.4f})")

# ========== Step 8: V3 物理特性分析 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 8】V3 物理特性分析")
logger.info("=" * 120)

if v3_results:
    logger.info("\nV3 检索结果物理分析:")
    logger.info(f"{'Rank':<6} {'场景':<15} {'温度':<10} {'记忆强度':<12} {'双曲距离':<12}")
    logger.info("-" * 80)

    for result in v3_results:
        chunk_type = chunk_types.get(chunks.index(result['text']), "Unknown")
        logger.info(f"{result['rank']:<6} {chunk_type:<15} "
                   f"{result['temperature']:<10.4f} "
                   f"{result['memory_strength']:<12.4f} "
                   f"{result['hyperbolic_distance']:<12.4f}")

# ========== Step 9: 对比分析 ==========
logger.info("\n" + "=" * 120)
logger.info("【Step 9】对比分析")
logger.info("=" * 120)

# 辅助函数
def get_rank(results, chunk_idx):
    for result in results:
        if result.get('text') == chunks[chunk_idx]:
            return result['rank']
    return None

# 排名对比
all_modes = [
    ("HippoRAG", hipporag_results),
    ("HyperAmy V3", v3_results),
    ("GraphFusion", fusion_results)
]

logger.info("\n【排名对比】")
logger.info(f"{'检索方式':<15} {'情感对峙':<12} {'东方哲学':<12} {'拒绝葡萄干':<12} {'药丸场景':<12} {'早餐场景':<12}")
logger.info("-" * 100)

for mode_name, results in all_modes:
    row_str = f"{mode_name:<15} "
    for chunk_idx in range(5):
        rank = get_rank(results, chunk_idx)
        rank_str = str(rank) if rank else "-"
        row_str += f"{rank_str:<12}"
    logger.info(row_str)

# 性能对比
logger.info("\n【性能对比】")
logger.info(f"{'检索方式':<15} {'时间(s)':<15} {'相对倍数':<15} {'吞吐量(query/s)':<20}")
logger.info("-" * 80)

# V3 时间分解
logger.info(f"V3 时间分解: 查询处理={v3_query_time:.2f}s + 检索={v3_time:.2f}s = 总计={v3_query_time + v3_time:.2f}s")

times = [
    ("HippoRAG", hipporag_time),
    ("HyperAmy V3", v3_time + v3_query_time),  # 包含查询处理时间
    ("GraphFusion", fusion_time)
]

min_time = min(t[1] for t in times if t[1] > 0)
for mode_name, time_cost in times:
    if time_cost > 0:
        relative = time_cost / min_time
        throughput = 1.0 / time_cost
        logger.info(f"{mode_name:<15} {time_cost:<15.2f} {relative:<15.2f}x {throughput:<20.4f}")
    else:
        logger.info(f"{mode_name:<15} {'N/A':<15} {'N/A':<15} {'N/A':<20}")

# 准确性分析
logger.info("\n【准确性分析】")
logger.info("正确答案: 东方哲学（chunk 3）应排在第1位")

for mode_name, results in all_modes:
    correct_rank = get_rank(results, 3)  # chunk 3 是正确答案
    if correct_rank:
        score = 6 - correct_rank  # 第1名=5分，第5名=1分
        logger.info(f"  {mode_name}: 正确答案排名={correct_rank}, 得分={score}/5")
    else:
        logger.info(f"  {mode_name}: 正确答案未找到, 得分=0/5")

# MRR（Mean Reciprocal Rank）计算
logger.info("\n【MRR（平均倒数排名）分析】")
for mode_name, results in all_modes:
    reciprocal_ranks = []
    for chunk_idx in range(5):
        rank = get_rank(results, chunk_idx)
        if rank:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    logger.info(f"  {mode_name}: MRR = {mrr:.4f}")

# V3 特性分析
if v3_results:
    logger.info("\n【V3 特有分析】")
    avg_memory_strength = sum(r['memory_strength'] for r in v3_results) / len(v3_results)
    avg_hyperbolic_dist = sum(r['hyperbolic_distance'] for r in v3_results) / len(v3_results)
    logger.info(f"  平均记忆强度: {avg_memory_strength:.4f}")
    logger.info(f"  平均双曲距离: {avg_hyperbolic_dist:.4f}")
    logger.info(f"  检索到的粒子数量: {len(v3_results)}")

# ========== 完成 ==========
logger.info("\n" + "=" * 120)
logger.info("✓ V3 对比测试完成！")
logger.info("=" * 120)
logger.info(f"\n日志已保存到: {log_file}")
