#!/usr/bin/env python3
"""
Amygdala 检索失败诊断脚本

分析为什么 Amygdala 双曲检索返回空结果
"""
import os
import sys
import time

# 设置环境变量
from llm.config import API_KEY, BASE_URL, DEFAULT_EMBEDDING_MODEL, API_URL_EMBEDDINGS
os.environ["OPENAI_API_KEY"] = API_KEY

from workflow import Amygdala
from poincare.retrieval import HyperAmyRetrieval
from ods import ChromaClient

print("=" * 80)
print("Amygdala 检索失败诊断")
print("=" * 80)

# 初始化 Amygdala
print("\n[1] 初始化 Amygdala...")
amygdala = Amygdala(
    save_dir='./test_graph_fusion_amygdala_db',
    particle_collection_name='fusion_particles',
    conversation_namespace='fusion',
    auto_link_particles=False
)

print(f"✓ Amygdala 初始化成功")
print(f"  - 粒子总数: {len(amygdala.particle_to_conversation)}")
print(f"  - 对话总数: {len(amygdala.conversation_store.get_all_ids())}")

# 检查粒子存储
print(f"\n[2] 检查粒子存储...")
storage = amygdala.particle_storage
collection = storage.ods_client.client.get_collection(name=storage.collection.name)
doc_count = collection.count()
print(f"✓ ChromaDB 集合: {storage.collection.name}")
print(f"  - 文档数: {doc_count}")

if doc_count == 0:
    print("❌ 错误：ChromaDB 集合为空！")
    sys.exit(1)

# 测试查询
print(f"\n[3] 生成查询粒子...")
query = "Why did the Count refuse the grapes?"
query_particles = amygdala.particle.process(
    text=query,
    text_id=f'test_query_{int(time.time())}'
)

print(f"✓ 查询粒子生成成功")
print(f"  - 粒子数: {len(query_particles)}")

if not query_particles:
    print("❌ 错误：未生成查询粒子")
    sys.exit(1)

query_particle = query_particles[0]
print(f"  - 实体: {query_particle.entity}")
print(f"  - 向量形状: {query_particle.emotion_vector.shape}")
print(f"  - 速度: {query_particle.speed:.4f}")
print(f"  - 温度: {query_particle.temperature:.4f}")

# 测试 ChromaDB 向量检索
print(f"\n[4] 测试 ChromaDB 向量检索...")
query_vec = ChromaClient.normalize_vector(query_particle.emotion_vector)

# 尝试不同的 n_results
for n_results in [5, 10, 20, 50, 100]:
    results = storage.ods_client.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        include=['metadatas', 'embeddings', 'documents']
    )

    returned_count = len(results['ids'][0]) if results['ids'] else 0
    print(f"  n_results={n_results:3d}: 返回 {returned_count} 个结果")

    if results['ids'] and results['ids'][0]:
        print(f"    前3个候选:")
        for i in range(min(3, returned_count)):
            pid = results['ids'][0][i]
            meta = results['metadatas'][0][i]
            entity = meta.get('entity', 'N/A')
            conv_id = meta.get('conversation_id', 'N/A')
            print(f"      {i+1}. {entity} (conv: {conv_id[:20]}...)")

# 测试双曲检索
print(f"\n[5] 测试双曲检索（不同 cone_width）...")
retriever = HyperAmyRetrieval(
    storage=storage,
    projector=amygdala.particle_projector
)

print(f"\n  预计算查询粒子状态...")
dynamic_query = amygdala.particle_projector.compute_state(
    vec=query_particle.emotion_vector,
    v=query_particle.speed,
    T=query_particle.temperature,
    born=query_particle.born,
    t_now=time.time(),
    weight=query_particle.weight
)

print(f"  - 计算成功")
print(f"  - 是否过期: {dynamic_query.get('is_expired', False)}")

# 测试不同的 cone_width
print(f"\n  测试不同 cone_width:")
for cone_width in [5, 10, 20, 50, 100]:
    print(f"\n  cone_width={cone_width}:")

    # Step 1: ChromaDB 查询
    try:
        results = storage.ods_client.query(
            query_embeddings=[query_vec],
            n_results=cone_width,
            include=['metadatas', 'embeddings']
        )

        if not results['ids'] or not results['ids'][0]:
            print(f"    ChromaDB 返回空结果")
            continue

        ids = results['ids'][0]
        metas = results['metadatas'][0]
        vecs = results['embeddings'][0]
        print(f"    ChromaDB 返回 {len(ids)} 个候选")

        # Step 2: 计算双曲距离
        scored_count = 0
        filtered_count = 0

        for pid, meta, vec in zip(ids, metas, vecs):
            try:
                # 计算双曲距离
                score = retriever._calculate_score_raw(
                    dynamic_query, vec, meta, time.time()
                )

                if score == float('inf'):
                    filtered_count += 1
                else:
                    scored_count += 1

            except Exception as e:
                print(f"    计算距离出错: {e}")
                continue

        print(f"    有效粒子: {scored_count}, 过期粒子: {filtered_count}")

        # Step 3: 完整检索
        search_results = retriever.search(
            query_entity=query_particle,
            top_k=5,
            cone_width=cone_width
        )
        print(f"    最终返回: {len(search_results)} 个结果")

        if search_results:
            print(f"    前3个结果:")
            for i, r in enumerate(search_results[:3]):
                entity = r.metadata.get('entity', 'N/A')
                print(f"      {i+1}. {entity} (distance: {r.score:.4f})")
        else:
            print(f"    ❌ 无结果")

    except Exception as e:
        print(f"    ❌ 异常: {e}")
        import traceback
        traceback.print_exc()

# 尝试直接访问数据库
print(f"\n[6] 直接访问数据库...")
try:
    all_docs = collection.get(include=['metadatas', 'embeddings'])
    print(f"  总文档数: {len(all_docs['ids'])}")

    if len(all_docs['ids']) > 0:
        print(f"  前3个文档:")
        for i in range(min(3, len(all_docs['ids']))):
            pid = all_docs['ids'][i]
            meta = all_docs['metadatas'][i]
            entity = meta.get('entity', 'N/A')
            print(f"    {i+1}. {entity}")
except Exception as e:
    print(f"  ❌ 访问失败: {e}")

print(f"\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
