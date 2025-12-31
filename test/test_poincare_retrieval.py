"""
测试：庞加莱球检索功能

测试流程：
1. 创建10个粒子并存储到庞加莱球
2. 每隔一段时间创建一个新粒子
3. 用新粒子状态检索之前的10个粒子
4. 验证返回结果按距离排序
"""
import time
import numpy as np
import logging
from pathlib import Path
from typing import List

from particle import ParticleEntity
from poincare import (
    ParticleProjector,
    HyperAmyStorage,
    HyperAmyRetrieval,
    SearchResult
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_particle(entity_id: str, emotion_vector: np.ndarray, 
                        speed: float, temperature: float,
                        entity: str = None, text_id: str = None) -> ParticleEntity:
    """
    创建测试粒子
    
    Args:
        entity_id: 实体 ID
        emotion_vector: 情绪向量（numpy array，可以是未归一化的）
        speed: 初始速度/强度
        temperature: 初始温度
        entity: 实体名称（可选）
        text_id: 文本 ID（可选）
    
    Returns:
        ParticleEntity 对象（emotion_vector 会被归一化，weight 存储原始模长）
    """
    # 计算权重（原始情绪向量的模长）
    weight = float(np.linalg.norm(emotion_vector))
    
    # 归一化情绪向量（存储方向）
    if weight > 1e-9:
        normalized_vector = emotion_vector / weight
    else:
        normalized_vector = emotion_vector.copy()
        weight = 0.0
    
    return ParticleEntity(
        entity_id=entity_id,
        entity=entity if entity is not None else f"Entity_{entity_id}",
        text_id=text_id if text_id is not None else f"text_{entity_id}",
        emotion_vector=normalized_vector,  # 归一化后的方向向量
        weight=weight,  # 存储原始模长作为质量
        speed=speed,
        temperature=temperature,
        born=time.time()
    )


def test_poincare_retrieval():
    """
    测试庞加莱球检索功能
    
    1. 创建10个不同的粒子并存储
    2. 每隔一段时间创建一个新粒子
    3. 用新粒子检索之前的10个粒子
    4. 验证返回结果按距离排序
    """
    print("=" * 100)
    print("测试：庞加莱球检索功能")
    print("=" * 100)
    
    # collection_name 决定数据库路径（隐式）
    collection_name = "test_retrieval"
    expected_db_path = f"./hyperamy_db_{collection_name}"
    
    # 清理之前的测试数据（如果存在）
    if Path(expected_db_path).exists():
        import shutil
        shutil.rmtree(expected_db_path)
        logger.info(f"清理旧的测试数据库: {expected_db_path}")
    
    try:
        # ========== Step 1: 初始化组件 ==========
        print("\n【Step 1】初始化组件...")
        projector = ParticleProjector(curvature=1.0, scaling_factor=2.0, max_radius=100.0)
        storage = HyperAmyStorage(collection_name=collection_name)
        retrieval = HyperAmyRetrieval(storage, projector)
        print("✓ 组件初始化成功")
        print(f"  - 投影器: curvature={projector.c}, scaling_factor={projector.scaling_factor}, max_radius={projector.max_radius}")
        print(f"  - Collection: {collection_name}")
        
        # ========== Step 2: 创建10个不同的粒子 ==========
        print("\n【Step 2】创建10个不同的粒子...")
        
        # 创建10个不同情绪和属性的粒子
        base_vectors = [
            np.array([0.9, 0.1, 0.0] + [0.0] * 27),  # 愤怒
            np.array([0.8, 0.2, 0.0] + [0.0] * 27),  # 愤怒（稍弱）
            np.array([0.0, 0.1, 0.9] + [0.0] * 27),  # 开心
            np.array([0.0, 0.2, 0.8] + [0.0] * 27),  # 开心（稍弱）
            np.array([0.5, 0.5, 0.0] + [0.0] * 27),  # 中性-愤怒
            np.array([0.0, 0.5, 0.5] + [0.0] * 27),  # 中性-开心
            np.array([0.3, 0.3, 0.3] + [0.0] * 27),  # 中性
            np.array([0.7, 0.1, 0.2] + [0.0] * 27),  # 混合情绪1
            np.array([0.2, 0.1, 0.7] + [0.0] * 27),  # 混合情绪2
            np.array([0.6, 0.2, 0.2] + [0.0] * 27),  # 混合情绪3
        ]
        
        stored_particles = []
        t_base = time.time()
        
        for i, vec in enumerate(base_vectors):
            particle = create_test_particle(
                entity_id=f"particle_{i+1:02d}",
                entity=f"Entity_{i+1:02d}",
                text_id=f"text_{i+1:02d}",
                emotion_vector=vec,
                speed=0.3 + i * 0.05,  # 速度从0.3到0.75
                temperature=0.4 + i * 0.03  # 温度从0.4到0.67
            )
            # 设置相同的生成时间（模拟同时创建）
            particle.born = t_base
            stored_particles.append(particle)
        
        print(f"✓ 创建了 {len(stored_particles)} 个粒子:")
        for i, p in enumerate(stored_particles, 1):
            print(f"  {i:2d}. {p.entity_id}: speed={p.speed:.2f}, T={p.temperature:.2f}, weight={p.weight:.4f}")
        
        # ========== Step 3: 存储10个粒子到庞加莱球 ==========
        print("\n【Step 3】存储10个粒子到庞加莱球...")
        storage.upsert_entities(entities=stored_particles, links_map=None)
        print(f"✓ 成功存储 {len(stored_particles)} 个粒子")
        
        # ========== Step 4: 每隔一段时间创建新粒子并检索 ==========
        print("\n【Step 4】每隔一段时间创建新粒子并检索...")
        print("=" * 100)
        
        # 定义检索时间点（秒）
        retrieval_intervals = [0, 5, 10, 20, 50]
        
        for interval in retrieval_intervals:
            print(f"\n--- 时间点: {interval} 秒后 ---")
            
            # 等待到指定时间点
            if interval > 0:
                time.sleep(interval - (time.time() - t_base) if interval > (time.time() - t_base) else 0)
            
            t_query = time.time()
            elapsed_time = t_query - t_base
            
            # 创建一个新的查询粒子（与第一个粒子相似，但速度不同）
            query_vec = base_vectors[0].copy() * 0.8  # 与particle_01相似但稍弱
            query_particle = create_test_particle(
                entity_id=f"query_{interval}s",
                entity="QueryEntity",
                text_id=f"query_text_{interval}s",
                emotion_vector=query_vec,
                speed=0.5,  # 中等速度
                temperature=0.5
            )
            query_particle.born = t_base  # 与存储粒子同时生成
            
            print(f"\n查询粒子信息:")
            print(f"  - ID: {query_particle.entity_id}")
            print(f"  - 速度: {query_particle.speed:.2f}")
            print(f"  - 温度: {query_particle.temperature:.2f}")
            print(f"  - 质量: {query_particle.weight:.4f}")
            print(f"  - 已过去时间: {elapsed_time:.2f} 秒")
            
            # 执行检索
            results = retrieval.search(
                query_entity=query_particle,
                top_k=10,  # 检索所有10个粒子
                cone_width=20,  # 扩大搜索范围
                max_neighbors=10,
                neighbor_penalty=1.1
            )
            
            print(f"\n检索结果（找到 {len(results)} 个粒子，按距离排序）:")
            print("-" * 100)
            print(f"{'排名':<6} {'ID':<15} {'距离':<12} {'速度':<10} {'温度':<10} {'质量':<10} {'匹配类型':<12} {'状态':<10}")
            print("-" * 100)
            
            # 验证结果是否按距离排序
            distances = [r.score for r in results]
            is_sorted = all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
            
            for rank, result in enumerate(results, 1):
                weight = result.metadata.get('weight', 'N/A')
                speed = result.metadata.get('v', 'N/A')
                temp = result.metadata.get('T', 'N/A')
                match_type = result.match_type
                
                # 计算粒子当前状态
                if weight != 'N/A':
                    try:
                        state = projector.compute_state(
                            vec=np.array(result.vector),
                            v=result.metadata['v'],
                            T=result.metadata['T'],
                            born=result.metadata['born'],
                            t_now=t_query,
                            weight=weight
                        )
                        is_expired = state.get('is_expired', False)
                        status = "已消失" if is_expired else "正常"
                    except:
                        status = "未知"
                else:
                    status = "未知"
                
                print(f"{rank:<6} {result.id:<15} {result.score:<12.6f} {speed:<10} {temp:<10} {weight:<10} {match_type:<12} {status:<10}")
            
            print("-" * 100)
            print(f"排序验证: {'✓ 正确排序' if is_sorted else '✗ 排序错误'}")
            
            # 显示距离分布统计
            if len(results) > 0:
                min_dist = min(distances)
                max_dist = max(distances)
                avg_dist = sum(distances) / len(distances)
                print(f"\n距离统计:")
                print(f"  - 最小距离: {min_dist:.6f}")
                print(f"  - 最大距离: {max_dist:.6f}")
                print(f"  - 平均距离: {avg_dist:.6f}")
                
                # 显示最相似的3个粒子
                print(f"\n最相似的3个粒子:")
                for rank, result in enumerate(results[:3], 1):
                    print(f"  {rank}. {result.id} (距离: {result.score:.6f})")
            
            print("\n" + "=" * 100)
        
        # ========== Step 5: 验证检索功能 ==========
        print("\n【Step 5】验证检索功能...")
        print("-" * 100)
        
        # 创建一个与某个存储粒子非常相似的查询粒子
        print("\n测试1: 查询与 particle_01 非常相似的粒子")
        similar_query = create_test_particle(
            entity_id="similar_query",
            emotion_vector=base_vectors[0] * 0.95,  # 95%相似
            speed=0.35,  # 接近 particle_01 的速度
            temperature=0.43
        )
        similar_query.born = t_base
        
        results = retrieval.search(
            query_entity=similar_query,
            top_k=5,
            cone_width=20,
            max_neighbors=5,
            neighbor_penalty=1.1
        )
        
        print(f"找到 {len(results)} 个结果:")
        for rank, result in enumerate(results, 1):
            print(f"  {rank}. {result.id}: 距离={result.score:.6f}")
        
        # 验证最相似的是 particle_01
        if len(results) > 0:
            top_result = results[0]
            is_correct = top_result.id == "particle_01"
            print(f"\n验证: 最相似的粒子是 particle_01? {'✓ 是' if is_correct else '✗ 否（实际是 ' + top_result.id + '）'}")
        
        # 测试2: 查询与存储粒子完全不同的粒子
        print("\n测试2: 查询与存储粒子完全不同的粒子")
        different_query = create_test_particle(
            entity_id="different_query",
            emotion_vector=np.array([0.0, 0.0, 1.0] + [0.0] * 27),  # 完全不同的方向
            speed=1.0,
            temperature=1.0
        )
        different_query.born = t_base
        
        results = retrieval.search(
            query_entity=different_query,
            top_k=5,
            cone_width=20,
            max_neighbors=5,
            neighbor_penalty=1.1
        )
        
        print(f"找到 {len(results)} 个结果:")
        for rank, result in enumerate(results, 1):
            print(f"  {rank}. {result.id}: 距离={result.score:.6f}")
        
        print("\n" + "=" * 100)
        print("✓ 检索功能测试完成！")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # 清理测试数据（可选）
        # import shutil
        # if Path(expected_db_path).exists():
        #     shutil.rmtree(expected_db_path)
        #     print(f"\n清理测试数据库: {expected_db_path}")
        pass


if __name__ == "__main__":
    test_poincare_retrieval()

