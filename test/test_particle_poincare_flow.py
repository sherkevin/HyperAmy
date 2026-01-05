"""
测试：粒子创建 -> 存储 -> 查询完整流程

测试流程：
1. 创建粒子（ParticleEntity，初始状态）
2. 送入庞加莱球（存储）
3. 查看双曲空间状态
4. 等待一段时间
5. 查询粒子状态
"""
import time
import numpy as np
import logging
from pathlib import Path

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


def test_particle_lifecycle():
    """
    测试粒子的完整生命周期：
    1. 创建粒子（初始状态）
    2. 存储到庞加莱球
    3. 查看双曲空间状态
    4. 等待一段时间
    5. 查询粒子状态
    """
    print("=" * 80)
    print("测试：粒子创建 -> 存储 -> 查询完整流程")
    print("=" * 80)
    
    # collection_name 决定数据库路径（隐式）
    collection_name = "test_particles"
    # 数据库路径由 collection_name 隐式决定（在 HyperAmyStorage 中自动生成）
    # 预期路径: ./hyperamy_db_test_particles
    
    # 清理之前的测试数据（如果存在）
    expected_db_path = f"./hyperamy_db_{collection_name}"
    if Path(expected_db_path).exists():
        import shutil
        shutil.rmtree(expected_db_path)
        logger.info(f"清理旧的测试数据库: {expected_db_path}")
    
    try:
        # ========== Step 1: 初始化组件 ==========
        print("\n【Step 1】初始化组件...")
        projector = ParticleProjector(curvature=1.0, scaling_factor=2.0, max_radius=100.0)
        print(f"  - 最大半径: {projector.max_radius}")
        # 不提供 persist_path，由 collection_name 隐式决定数据库路径
        storage = HyperAmyStorage(collection_name=collection_name)
        retrieval = HyperAmyRetrieval(storage, projector)
        print("✓ 组件初始化成功")
        print(f"  - 投影器: curvature={projector.c}, scaling_factor={projector.scaling_factor}")
        print(f"  - Collection: {collection_name}")
        # 获取实际使用的数据库路径
        actual_path = getattr(storage.ods_client.client, '_path', expected_db_path)
        print(f"  - 存储路径: {actual_path} (由 collection_name 隐式决定)")
        
        # ========== Step 2: 创建粒子（初始状态） ==========
        print("\n【Step 2】创建粒子（初始状态）...")
        
        # 创建几个测试粒子
        # 粒子 A: 愤怒情绪，高强度
        vec_anger = np.array([0.9, 0.1, 0.0, 0.0] + [0.0] * 26)  # 30维向量
        particle_a = create_test_particle(
            entity_id="particle_A",
            emotion_vector=vec_anger,
            speed=0.9,  # 高强度
            temperature=0.5
        )
        
        # 粒子 B: 愤怒情绪，低强度（A 的邻居）
        particle_b = create_test_particle(
            entity_id="particle_B",
            emotion_vector=vec_anger * 0.5,  # 相似方向，但强度不同
            speed=0.3,  # 低强度
            temperature=0.5
        )
        
        # 粒子 C: 开心情绪
        vec_happy = np.array([0.0, 0.1, 0.9, 0.0] + [0.0] * 26)
        particle_c = create_test_particle(
            entity_id="particle_C",
            emotion_vector=vec_happy,
            speed=0.7,
            temperature=0.6
        )
        
        print(f"✓ 创建了 3 个测试粒子:")
        print(f"  - {particle_a.entity_id}: 愤怒(高), speed={particle_a.speed:.2f}, T={particle_a.temperature:.2f}, weight={particle_a.weight:.4f}")
        print(f"  - {particle_b.entity_id}: 愤怒(低), speed={particle_b.speed:.2f}, T={particle_b.temperature:.2f}, weight={particle_b.weight:.4f}")
        print(f"  - {particle_c.entity_id}: 开心, speed={particle_c.speed:.2f}, T={particle_c.temperature:.2f}, weight={particle_c.weight:.4f}")
        
        # ========== Step 3: 查看初始状态的双曲空间坐标 ==========
        print("\n【Step 3】查看初始状态的双曲空间坐标...")
        t_now = time.time()
        
        state_a = projector.compute_state(
            vec=particle_a.emotion_vector,
            v=particle_a.speed,
            T=particle_a.temperature,
            born=particle_a.born,
            t_now=t_now,
            weight=particle_a.weight
        )
        
        print(f"✓ 粒子 A 的双曲空间状态:")
        print(f"  - 双曲坐标: {state_a['current_vector'][:3].tolist()}... (前3维)")
        print(f"  - 当前速度: {state_a['current_v']:.4f}")
        print(f"  - 当前温度: {state_a['current_T']:.4f}")
        print(f"  - 坐标范数: {np.linalg.norm(state_a['current_vector'].numpy()):.4f}")
        print(f"  - 距离原点: {state_a['distance_from_origin']:.4f}")
        print(f"  - 粒子质量: {particle_a.weight:.4f}")
        print(f"  - 是否消失: {'是' if state_a.get('is_expired', False) else '否'}")
        
        # ========== Step 4: 存储粒子到庞加莱球 ==========
        print("\n【Step 4】存储粒子到庞加莱球...")
        
        # 构建链接关系（A 连接到 B）
        links_map = {
            "particle_A": ["particle_B"],
            "particle_B": ["particle_A"],
            "particle_C": []
        }
        
        # 批量存储
        storage.upsert_entities(
            entities=[particle_a, particle_b, particle_c],
            links_map=links_map
        )
        
        print("✓ 粒子存储成功")
        print(f"  - 存储了 {len([particle_a, particle_b, particle_c])} 个粒子")
        print(f"  - 链接关系: A <-> B")
        
        # ========== Step 5: 等待一段时间 ==========
        print("\n【Step 5】等待一段时间（模拟时间流逝）...")
        wait_time = 2.0  # 等待 2 秒
        print(f"  等待 {wait_time} 秒...")
        time.sleep(wait_time)
        
        t_query = time.time()
        elapsed_time = t_query - particle_a.born
        print(f"✓ 时间已过去 {elapsed_time:.2f} 秒")
        
        # ========== Step 6: 查询粒子状态 ==========
        print("\n【Step 6】查询粒子状态...")
        
        # 使用粒子 A 作为查询
        print(f"  使用 {particle_a.entity_id} 作为查询粒子...")
        
        results = retrieval.search(
            query_entity=particle_a,
            top_k=3,
            cone_width=10,
            max_neighbors=5,
            neighbor_penalty=1.1
        )
        
        print(f"✓ 查询完成，找到 {len(results)} 个结果:")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"  - ID: {result.id}")
            print(f"  - 匹配类型: {result.match_type}")
            print(f"  - 双曲距离: {result.score:.4f}")
            print(f"  - 元数据:")
            print(f"    * speed (v): {result.metadata.get('v', 'N/A')}")
            print(f"    * temperature (T): {result.metadata.get('T', 'N/A')}")
            print(f"    * born: {result.metadata.get('born', 'N/A')}")
            print(f"    * entity: {result.metadata.get('entity', 'N/A')}")
            print(f"    * text_id: {result.metadata.get('text_id', 'N/A')}")
            
            # 计算查询时的动态状态
            if result.id == particle_a.entity_id:
                # 这是查询粒子本身
                weight = result.metadata.get('weight', 1.0)
                dynamic_state = projector.compute_state(
                    vec=np.array(result.vector),
                    v=result.metadata['v'],
                    T=result.metadata['T'],
                    born=result.metadata['born'],
                    t_now=t_query,
                    weight=weight
                )
                print(f"  - 查询时的动态状态:")
                print(f"    * 双曲坐标: {dynamic_state['current_vector'][:3].tolist()}... (前3维)")
                print(f"    * 当前速度: {dynamic_state['current_v']:.4f}")
                print(f"    * 当前温度: {dynamic_state['current_T']:.4f}")
                print(f"    * 距离原点: {dynamic_state['distance_from_origin']:.4f}")
                print(f"    * 是否消失: {'是' if dynamic_state.get('is_expired', False) else '否'}")
        
        # ========== Step 7: 验证查询结果 ==========
        print("\n【Step 7】验证查询结果...")
        
        # 检查是否找到了粒子 A 本身
        found_a = any(r.id == "particle_A" for r in results)
        found_b = any(r.id == "particle_B" for r in results)
        
        print(f"✓ 验证结果:")
        print(f"  - 找到粒子 A: {'是' if found_a else '否'}")
        print(f"  - 找到粒子 B (邻居): {'是' if found_b else '否'}")
        
        # 验证双曲距离：A 到自己的距离应该最小（接近 0）
        if found_a:
            dist_to_self = next(r.score for r in results if r.id == "particle_A")
            print(f"  - A 到自己的距离: {dist_to_self:.6f} (应该接近 0)")
        
        # 验证邻居关系：B 应该通过链接被找到
        if found_b:
            b_result = next(r for r in results if r.id == "particle_B")
            print(f"  - B 的匹配类型: {b_result.match_type}")
            if b_result.match_type == 'neighbor':
                print(f"  - ✓ B 通过邻居链接被找到")
        
        print("\n" + "=" * 80)
        print("✓ 测试完成！")
        print("=" * 80)
        
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


def test_time_evolution():
    """
    测试时间演化：验证粒子状态随时间的变化
    """
    print("\n" + "=" * 80)
    print("测试：时间演化验证")
    print("=" * 80)
    
    projector = ParticleProjector(curvature=1.0, scaling_factor=2.0, max_radius=100.0)
    
    # 创建一个粒子
    vec = np.array([0.8, 0.2, 0.0] + [0.0] * 27)
    particle = create_test_particle(
        entity_id="test_time",
        emotion_vector=vec,
        speed=0.8,
        temperature=0.6
    )
    
    t_born = particle.born
    
    # 在不同时间点查询状态
    time_points = [0, 1, 5, 10, 50, 100]  # 秒
    
    print("\n时间演化状态:")
    print("-" * 80)
    print(f"{'时间点(s)':<12} {'当前速度':<12} {'当前温度':<12} {'距离原点':<12} {'坐标范数':<12} {'状态':<10}")
    print("-" * 80)
    
    for t_offset in time_points:
        t_now = t_born + t_offset
        state = projector.compute_state(
            vec=particle.emotion_vector,
            v=particle.speed,
            T=particle.temperature,
            born=t_born,
            t_now=t_now,
            weight=particle.weight
        )
        
        coord_norm = np.linalg.norm(state['current_vector'].numpy())
        distance = state['distance_from_origin']
        is_expired = state.get('is_expired', False)
        status = "已消失" if is_expired else "正常"
        print(f"{t_offset:<12} {state['current_v']:<12.4f} {state['current_T']:<12.4f} {distance:<12.4f} {coord_norm:<12.4f} {status:<10}")
    
    print("-" * 80)
    print("✓ 时间演化测试完成")
    print("  注意：")
    print("  - 当前 TimePhysics 实现为恒等映射，速度和温度不随时间变化")
    print("  - 但粒子距离原点会随时间增加：距离 = 初始距离 + 速度 × 时间")
    print("  - 当距离 >= 最大半径(100)时，粒子被认为已消失")


def test_particle_tracking():
    """
    测试粒子追踪：追踪某个粒子在不同时刻的完整状态信息
    
    验证：
    1. 距离随时间线性增加：距离 = 初始距离 + 速度 × 时间
    2. 距离增量 = 速度 × 时间差
    3. 输出粒子的完整信息，包括质量
    """
    print("\n" + "=" * 80)
    print("测试：粒子追踪 - 完整状态信息")
    print("=" * 80)
    
    projector = ParticleProjector(curvature=1.0, scaling_factor=2.0, max_radius=100.0)
    
    # 创建一个测试粒子
    vec = np.array([0.9, 0.1, 0.0, 0.0] + [0.0] * 26)  # 30维向量
    particle = create_test_particle(
        entity_id="tracked_particle",
        entity="TrackedEntity",
        text_id="text_tracked",
        emotion_vector=vec,
        speed=0.5,  # 速度 0.5 单位/秒
        temperature=0.7
    )
    
    print("\n【粒子初始信息】")
    print("-" * 80)
    print(f"  - 实体 ID: {particle.entity_id}")
    print(f"  - 实体名称: {particle.entity}")
    print(f"  - 文本 ID: {particle.text_id}")
    print(f"  - 情绪向量形状: {particle.emotion_vector.shape}")
    print(f"  - 情绪向量（前5维）: {particle.emotion_vector[:5].tolist()}")
    print(f"  - 粒子质量 (weight): {particle.weight:.6f}")
    print(f"  - 初始速度 (speed): {particle.speed:.6f}")
    print(f"  - 初始温度 (temperature): {particle.temperature:.6f}")
    print(f"  - 生成时间 (born): {particle.born:.6f}")
    
    # 计算初始距离
    initial_distance = particle.weight * projector.scaling_factor
    print(f"  - 初始距离 (weight × scaling_factor): {initial_distance:.6f}")
    print("-" * 80)
    
    t_born = particle.born
    
    # 追踪时间点（秒）
    time_points = [0, 1, 2, 5, 10, 20, 50, 100, 150, 200]
    
    print("\n【粒子状态追踪】")
    print("=" * 120)
    print(f"{'时间点(s)':<12} {'距离原点':<12} {'距离增量':<12} {'预期增量':<12} {'坐标范数':<12} {'速度':<10} {'温度':<10} {'状态':<10}")
    print("=" * 120)
    
    prev_distance = None
    prev_time = None
    
    for t_offset in time_points:
        t_now = t_born + t_offset
        state = projector.compute_state(
            vec=particle.emotion_vector,
            v=particle.speed,
            T=particle.temperature,
            born=t_born,
            t_now=t_now,
            weight=particle.weight
        )
        
        distance = state['distance_from_origin']
        coord_norm = np.linalg.norm(state['current_vector'].numpy())
        is_expired = state.get('is_expired', False)
        status = "已消失" if is_expired else "正常"
        
        # 计算距离增量
        if prev_distance is not None:
            distance_delta = distance - prev_distance
            time_delta = t_offset - prev_time
            expected_delta = particle.speed * time_delta
            delta_diff = abs(distance_delta - expected_delta)
            delta_status = "✓" if delta_diff < 1e-6 else f"✗({delta_diff:.6f})"
            distance_delta_str = f"{distance_delta:.6f} {delta_status}"
            expected_delta_str = f"{expected_delta:.6f}"
        else:
            distance_delta_str = "-"
            expected_delta_str = "-"
        
        print(f"{t_offset:<12} {distance:<12.6f} {distance_delta_str:<12} {expected_delta_str:<12} {coord_norm:<12.6f} {state['current_v']:<10.6f} {state['current_T']:<10.6f} {status:<10}")
        
        prev_distance = distance
        prev_time = t_offset
    
    print("=" * 120)
    
    # 验证距离计算的合理性
    print("\n【距离计算验证】")
    print("-" * 80)
    
    # 选择几个时间点进行详细验证
    test_times = [0, 5, 10, 50]
    print(f"{'时间点(s)':<12} {'计算距离':<15} {'公式计算':<15} {'差值':<15} {'状态':<10}")
    print("-" * 80)
    
    for t_offset in test_times:
        t_now = t_born + t_offset
        state = projector.compute_state(
            vec=particle.emotion_vector,
            v=particle.speed,
            T=particle.temperature,
            born=t_born,
            t_now=t_now,
            weight=particle.weight
        )
        
        computed_distance = state['distance_from_origin']
        # 公式：距离 = 初始距离 + 速度 × 时间
        formula_distance = initial_distance + particle.speed * t_offset
        diff = abs(computed_distance - formula_distance)
        is_expired = state.get('is_expired', False)
        status = "已消失" if is_expired else "正常"
        
        match_status = "✓" if diff < 1e-6 else f"✗"
        print(f"{t_offset:<12} {computed_distance:<15.6f} {formula_distance:<15.6f} {diff:<15.6e} {match_status} {status:<10}")
    
    print("-" * 80)
    
    # 输出完整粒子信息摘要
    print("\n【粒子完整信息摘要】")
    print("-" * 80)
    print(f"实体信息:")
    print(f"  - entity_id: {particle.entity_id}")
    print(f"  - entity: {particle.entity}")
    print(f"  - text_id: {particle.text_id}")
    print(f"\n物理属性:")
    print(f"  - weight (质量): {particle.weight:.6f} (初始情绪向量模长)")
    print(f"  - speed (速度): {particle.speed:.6f}")
    print(f"  - temperature (温度): {particle.temperature:.6f}")
    print(f"  - born (生成时间): {particle.born:.6f}")
    print(f"\n空间属性:")
    print(f"  - emotion_vector (归一化方向向量): shape={particle.emotion_vector.shape}")
    print(f"  - emotion_vector 前5维: {particle.emotion_vector[:5].tolist()}")
    print(f"  - emotion_vector 模长: {np.linalg.norm(particle.emotion_vector):.6f} (应该为 1.0)")
    print(f"\n计算参数:")
    print(f"  - 初始距离: {initial_distance:.6f} = weight({particle.weight:.6f}) × scaling_factor({projector.scaling_factor})")
    print(f"  - 距离公式: distance(t) = {initial_distance:.6f} + {particle.speed:.6f} × t")
    print(f"  - 最大半径: {projector.max_radius}")
    print("-" * 80)
    
    print("\n✓ 粒子追踪测试完成")
    print("  验证结果:")
    print("  - 距离随时间线性增加：距离 = 初始距离 + 速度 × 时间")
    print("  - 距离增量 = 速度 × 时间差")
    print("  - 当距离 >= 最大半径时，粒子标记为已消失")


if __name__ == "__main__":
    # 运行完整生命周期测试
    test_particle_lifecycle()
    
    # 运行时间演化测试
    test_time_evolution()
    
    # 运行粒子追踪测试
    test_particle_tracking()

