"""
H-Mem System V3 端到端集成测试

测试完整的 H-Mem 系统流程：
1. 粒子创建（从语义嵌入）
2. 物理属性计算（mass, temperature, initial_radius）
3. 存储（写入数据库）
4. 时间演化（模拟经过时间）
5. 检索（三步检索流程）
6. 验证物理行为（质量影响衰减、温度调制检索等）
"""

import numpy as np
import tempfile
import shutil
import math

from poincare.storage import HyperAmyStorage
from poincare.retrieval import InMemoryRetrieval, RetrievalConfig, create_candidate
from poincare.physics import PhysicsEngine
from particle.properties import PropertyCalculator


class TestHMemIntegration:
    """H-Mem 系统端到端集成测试"""

    def test_full_retrieval_pipeline(self):
        """测试完整的检索流程"""
        print("\n    测试完整检索流程...")

        # 创建物理引擎和检索器
        config = RetrievalConfig(
            semantic_threshold=0.7,  # 语义相似度阈值
            retrieval_beta=1.0,      # 温度调制系数
            gamma=1.0                # 衰减常数
        )
        retrieval = InMemoryRetrieval(config=config)
        t_start = 0.0

        # 创建不同语义方向的粒子
        particles = [
            # 相同方向，高质量（创伤记忆）
            create_candidate("trauma", np.array([1.0, 0.0]), mass=100.0, temperature=0.5, initial_radius=2.0, created_at=t_start),
            # 相同方向，低质量（琐碎记忆）
            create_candidate("trivia", np.array([1.0, 0.0]), mass=0.1, temperature=2.0, initial_radius=0.5, created_at=t_start),
            # 相似方向
            create_candidate("similar", np.array([0.9, 0.1]), mass=1.0, temperature=1.0, initial_radius=1.0, created_at=t_start),
            # 不同方向
            create_candidate("different", np.array([0.0, 1.0]), mass=1.0, temperature=1.0, initial_radius=1.0, created_at=t_start),
        ]

        retrieval.add_particles(particles)

        # 查询：与 [1, 0] 相同方向
        results = retrieval.search(
            query_direction=np.array([1.0, 0.0]),
            query_mass=1.0,
            query_temperature=1.0,
            query_initial_radius=1.0,
            top_k=10,
            t_now=t_start
        )

        # 验证：
        # 1. 应该返回 trauma, trivia, similar（语义相似）
        # 2. 不应该返回 different（语义不相似）
        result_ids = [r.id for r in results]
        assert "different" not in result_ids, "语义不相似的粒子不应被检索"
        assert len(results) >= 2, "应该至少返回2个相似粒子"

        print(f"      ✓ 检索到 {len(results)} 个粒子")

    def test_time_decay_affects_retrieval(self):
        """测试时间衰减影响检索排序"""
        print("\n    测试时间衰减影响检索排序...")

        retrieval = InMemoryRetrieval(config=RetrievalConfig(gamma=1.0, semantic_threshold=0.9))
        t_start = 0.0

        # 两个相同方向的粒子，但质量不同
        particles = [
            create_candidate("heavy", np.array([1.0, 0.0]), mass=10.0, temperature=1.0, initial_radius=2.0, created_at=t_start),
            create_candidate("light", np.array([1.0, 0.0]), mass=0.5, temperature=1.0, initial_radius=2.0, created_at=t_start),
        ]
        retrieval.add_particles(particles)

        # 立即检索：heavy 排在前面（因为质量大 → 衰减慢 → 保持更接近查询点）
        results_t0 = retrieval.search(
            query_direction=np.array([1.0, 0.0]),
            query_mass=1.0,
            query_temperature=1.0,
            query_initial_radius=1.0,
            top_k=10,
            t_now=t_start
        )

        # 经过较短时间后检索（避免 light 被完全遗忘）
        t_later = 2.0
        results_t5 = retrieval.search(
            query_direction=np.array([1.0, 0.0]),
            query_mass=1.0,
            query_temperature=1.0,
            query_initial_radius=1.0,
            top_k=10,
            t_now=t_later
        )

        # heavy 应该始终排在前面
        assert results_t0[0].id == "heavy", "初始时刻高质量粒子应排在前面"
        assert results_t5[0].id == "heavy", "一段时间后高质量粒子仍应排在前面"

        # light 粒子的分数应该随时间下降更多
        heavy_score_t0 = next(r.score for r in results_t0 if r.id == "heavy")
        heavy_score_t5 = next(r.score for r in results_t5 if r.id == "heavy")
        light_score_t0 = next(r.score for r in results_t0 if r.id == "light")

        # 检查 light 是否还存在（可能被遗忘）
        light_results_t5 = [r for r in results_t5 if r.id == "light"]
        if light_results_t5:
            light_score_t5 = light_results_t5[0].score
            light_decay = (light_score_t0 - light_score_t5) / light_score_t0
            heavy_decay = (heavy_score_t0 - heavy_score_t5) / heavy_score_t0
            assert light_decay > heavy_decay, "低质量粒子的分数衰减应该更快"
            print(f"      ✓ Heavy 衰减: {heavy_decay:.2%}, Light 衰减: {light_decay:.2%}")
        else:
            # light 被遗忘，也是预期行为
            print(f"      ✓ Heavy 保持检索，Light 已被遗忘（符合预期）")

    def test_temperature_modulates_retrieval(self):
        """测试温度调制检索行为"""
        print("\n    测试温度调制检索行为...")

        # 高 beta 表示温度对检索的影响更大
        retrieval = InMemoryRetrieval(config=RetrievalConfig(retrieval_beta=2.0, semantic_threshold=0.0))
        t_start = 0.0

        # 两个位置相同但温度不同的粒子
        # 高温 = 模糊记忆，更容易被检索（距离惩罚小）
        # 低温 = 清晰记忆，需要精确匹配（距离惩罚大）
        particles = [
            create_candidate("fuzzy", np.array([0.8, 0.2]), mass=1.0, temperature=5.0, initial_radius=1.0, created_at=t_start),
            create_candidate("clear", np.array([0.8, 0.2]), mass=1.0, temperature=0.2, initial_radius=1.0, created_at=t_start),
        ]
        retrieval.add_particles(particles)

        # 查询方向略有不同
        results = retrieval.search(
            query_direction=np.array([1.0, 0.0]),  # 与候选方向有差异
            query_mass=1.0,
            query_temperature=1.0,
            query_initial_radius=1.0,
            top_k=10,
            t_now=t_start
        )

        # 高温（模糊）记忆应该排在前面，因为温度调制降低了距离惩罚
        assert results[0].id == "fuzzy", "高温（模糊）记忆应该更容易被检索"

        print(f"      ✓ Fuzzy 排第1, Clear 排第2（温度调制生效）")

    def test_forgetting_mechanism(self):
        """测试遗忘机制"""
        print("\n    测试遗忘机制...")

        retrieval = InMemoryRetrieval(
            config=RetrievalConfig(
                gamma=1.0,
                forgetting_threshold=0.1,
                semantic_threshold=0.9
            )
        )
        t_start = 0.0

        # 创建一个小质量粒子（会快速遗忘）
        retrieval.add_particle(create_candidate(
            "ephemeral",
            direction=np.array([1.0, 0.0]),
            mass=0.1,  # 小质量，快速衰减
            temperature=1.0,
            initial_radius=1.0,
            created_at=t_start
        ))

        # 初始时刻可以检索到
        results_t0 = retrieval.search(
            query_direction=np.array([1.0, 0.0]),
            top_k=10,
            t_now=t_start
        )
        assert len(results_t0) == 1, "初始时刻应该能检索到"

        # 足够长时间后，粒子被遗忘
        results_t100 = retrieval.search(
            query_direction=np.array([1.0, 0.0]),
            top_k=10,
            t_now=100.0
        )
        assert len(results_t100) == 0, "遗忘的粒子不应被检索"

        print(f"      ✓ 粒子在长时间后被正确遗忘")

    def test_property_calculation_integration(self):
        """测试属性计算器的集成"""
        print("\n    测试属性计算器集成...")

        calc = PropertyCalculator()

        # 测试质量计算：m = α*I + β*log(1 + κ)
        mass_high_intensity = calc.compute_mass(intensity=0.9, purity=0.8)
        mass_low_intensity = calc.compute_mass(intensity=0.1, purity=0.8)
        assert mass_high_intensity > mass_low_intensity, "高强度应该产生更大质量"

        mass_high_purity = calc.compute_mass(intensity=0.5, purity=0.9)
        mass_low_purity = calc.compute_mass(intensity=0.5, purity=0.1)
        assert mass_high_purity > mass_low_purity, "高纯度应该产生更大质量"

        # 测试温度计算：T = T0 / κ
        temp_high_purity = calc.compute_temperature(purity=0.9)
        temp_low_purity = calc.compute_temperature(purity=0.1)
        assert temp_high_purity < temp_low_purity, "高纯度应该产生更低温度"

        # 测试初始半径计算：R0 = scale * m
        radius = calc.compute_initial_radius(mass=2.0)
        assert radius > 0, "初始半径应该为正"

        print(f"      ✓ 质量、温度、半径计算正确")

    def test_physics_engine_integration(self):
        """测试物理引擎的集成"""
        print("\n    测试物理引擎集成...")

        physics = PhysicsEngine(curvature=1.0, gamma=1.0)

        # 测试引力时间膨胀：大质量 → 衰减慢
        direction = np.array([1.0, 0.0])
        state_heavy = physics.compute_state(
            direction=direction,
            mass=100.0,
            temperature=1.0,
            initial_radius=2.0,
            created_at=0.0,
            t_now=10.0
        )
        state_light = physics.compute_state(
            direction=direction,
            mass=0.1,
            temperature=1.0,
            initial_radius=2.0,
            created_at=0.0,
            t_now=10.0
        )

        # 大质量粒子的半径应该更大（衰减更慢）
        assert state_heavy.hyperbolic_radius > state_light.hyperbolic_radius, "大质量粒子衰减更慢"
        assert state_heavy.memory_strength > state_light.memory_strength, "大质量粒子保持更高强度"

        # 小质量粒子应该被遗忘
        assert not state_heavy.is_forgotten, "大质量粒子不应被遗忘"
        assert state_light.is_forgotten, "小质量粒子应该被遗忘"

        print(f"      ✓ Heavy R={state_heavy.hyperbolic_radius:.4f}, Light R={state_light.hyperbolic_radius:.6f}")

    def test_storage_integration(self):
        """测试存储层的集成"""
        print("\n    测试存储层集成...")

        temp_dir = tempfile.mkdtemp()
        try:
            storage = HyperAmyStorage(persist_path=temp_dir, collection_name="test_integration")

            # 创建并存储粒子
            particles = [
                create_candidate(f"p{i}", np.array([1.0, 0.0]), mass=1.0, temperature=1.0, initial_radius=1.0, created_at=0.0)
                for i in range(3)
            ]
            storage.upsert_candidates(particles)

            # 验证存储
            assert storage.count_candidates() == 3, "应该存储3个粒子"

            # 验证检索
            retrieved = storage.get_all_candidates()
            assert len(retrieved) == 3, "应该能检索到3个粒子"

            # 验证向量查询
            results = storage.query_by_direction(
                direction=np.array([1.0, 0.0]),
                n_results=2
            )
            assert len(results) <= 2, "查询应该限制返回数量"

            print(f"      ✓ 存储和查询正常工作")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_complete_workflow(self):
        """测试完整的工作流程：创建 → 存储 → 时间演化 → 检索"""
        print("\n    测试完整工作流程...")

        temp_dir = tempfile.mkdtemp()
        try:
            # 1. 初始化组件
            storage = HyperAmyStorage(persist_path=temp_dir, collection_name="workflow_test")
            physics = PhysicsEngine(gamma=1.0)
            calc = PropertyCalculator()
            retrieval = InMemoryRetrieval(config=RetrievalConfig(semantic_threshold=0.7))

            t_start = 0.0

            # 2. 创建粒子（模拟从语义信息创建）
            semantic_direction = np.array([1.0, 0.0])

            # 计算物理属性
            intensity = 0.8
            purity = 0.7
            mass = calc.compute_mass(intensity, purity)
            temperature = calc.compute_temperature(purity)
            initial_radius = calc.compute_initial_radius(mass)

            particle = create_candidate(
                particle_id="semantic_memory",
                direction=semantic_direction,
                mass=mass,
                temperature=temperature,
                initial_radius=initial_radius,
                created_at=t_start
            )

            # 3. 存储到数据库
            storage.upsert_candidate(particle)

            # 4. 同时添加到内存检索器
            retrieval.add_particle(particle)

            # 5. 验证物理引擎计算
            state = physics.compute_state(
                direction=semantic_direction,
                mass=mass,
                temperature=temperature,
                initial_radius=initial_radius,
                created_at=t_start,
                t_now=t_start
            )
            assert state.hyperbolic_radius == initial_radius, "初始时刻半径应等于初始值"
            assert state.memory_strength == 1.0, "初始时刻记忆强度应为1"

            # 6. 模拟时间演化
            t_later = 5.0
            state_later = physics.compute_state(
                direction=semantic_direction,
                mass=mass,
                temperature=temperature,
                initial_radius=initial_radius,
                created_at=t_start,
                t_now=t_later
            )
            assert state_later.hyperbolic_radius < initial_radius, "半径应该随时间衰减"
            assert 0 < state_later.memory_strength < 1.0, "记忆强度应该在(0,1)之间"

            # 7. 检索测试
            results = retrieval.search(
                query_direction=semantic_direction,
                query_mass=1.0,
                query_temperature=1.0,
                query_initial_radius=1.0,
                top_k=10,
                t_now=t_later
            )
            assert len(results) == 1, "应该能检索到创建的粒子"

            # 8. 验证存储
            stored_particles = storage.get_all_candidates()
            assert len(stored_particles) == 1, "存储中应该有1个粒子"
            assert stored_particles[0].id == "semantic_memory", "粒子ID应该匹配"

            print(f"      ✓ 完整工作流程测试通过")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """运行所有集成测试"""
    print("\n" + "=" * 70)
    print("H-Mem System V3 - 端到端集成测试")
    print("=" * 70)

    test = TestHMemIntegration()

    print("\n【测试组 1】检索流程测试")
    test.test_full_retrieval_pipeline()

    print("\n【测试组 2】时间衰减测试")
    test.test_time_decay_affects_retrieval()

    print("\n【测试组 3】温度调制测试")
    test.test_temperature_modulates_retrieval()

    print("\n【测试组 4】遗忘机制测试")
    test.test_forgetting_mechanism()

    print("\n【测试组 5】属性计算器测试")
    test.test_property_calculation_integration()

    print("\n【测试组 6】物理引擎测试")
    test.test_physics_engine_integration()

    print("\n【测试组 7】存储层测试")
    test.test_storage_integration()

    print("\n【测试组 8】完整工作流程测试")
    test.test_complete_workflow()

    print("\n" + "=" * 70)
    print("✓ 所有集成测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
