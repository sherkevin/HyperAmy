"""
测试 poincare/retrieval.py 模块 (System V3)

验证检索系统的正确性。

测试内容：
1. 锥体语义过滤
2. 引力投影
3. 热力学采样
4. 完整检索流程
5. 温度调制效应
"""

import numpy as np
import time

from poincare.retrieval import (
    RetrievalConfig,
    CandidateParticle,
    RetrievalResult,
    HMemRetrieval,
    InMemoryRetrieval,
    create_candidate,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_RETRIEVAL_BETA,
)


class TestRetrievalConfig:
    """测试检索配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = RetrievalConfig()
        assert config.semantic_threshold == DEFAULT_SEMANTIC_THRESHOLD
        assert config.retrieval_beta == DEFAULT_RETRIEVAL_BETA
        assert config.curvature == 1.0
        assert config.gamma == 1.0

    def test_custom_config(self):
        """测试自定义配置"""
        config = RetrievalConfig(
            semantic_threshold=0.7,
            retrieval_beta=2.0,
            curvature=0.5,
            gamma=2.0
        )
        assert config.semantic_threshold == 0.7
        assert config.retrieval_beta == 2.0
        assert config.curvature == 0.5
        assert config.gamma == 2.0


class TestSemanticPruning:
    """测试锥体语义过滤"""

    def test_filters_by_similarity(self):
        """测试按相似度过滤"""
        retrieval = HMemRetrieval(
            config=RetrievalConfig(semantic_threshold=0.5)
        )

        # 创建候选粒子
        candidates = [
            CandidateParticle(
                id="p1",
                direction=np.array([1.0, 0.0, 0.0]),  # 相似度 = 1.0
                mass=1.0,
                temperature=1.0,
                initial_radius=1.0,
                created_at=0.0
            ),
            CandidateParticle(
                id="p2",
                direction=np.array([0.0, 1.0, 0.0]),  # 相似度 = 0.0
                mass=1.0,
                temperature=1.0,
                initial_radius=1.0,
                created_at=0.0
            ),
            CandidateParticle(
                id="p3",
                direction=np.array([0.707, 0.707, 0.0]),  # 相似度 ≈ 0.707
                mass=1.0,
                temperature=1.0,
                initial_radius=1.0,
                created_at=0.0
            ),
        ]

        query_dir = np.array([1.0, 0.0, 0.0])

        filtered = retrieval._semantic_pruning(candidates, query_dir)

        # p1 (相似度 1.0) 和 p3 (相似度 0.707) 应该保留
        assert len(filtered) == 2
        assert any(c.id == "p1" for c in filtered)
        assert any(c.id == "p3" for c in filtered)
        assert not any(c.id == "p2" for c in filtered)

    def test_empty_when_no_match(self):
        """测试没有匹配时返回空"""
        retrieval = HMemRetrieval(
            config=RetrievalConfig(semantic_threshold=0.9)
        )

        candidates = [
            CandidateParticle(
                id="p1",
                direction=np.array([0.0, 1.0, 0.0]),  # 相似度 = 0.0
                mass=1.0,
                temperature=1.0,
                initial_radius=1.0,
                created_at=0.0
            ),
        ]

        query_dir = np.array([1.0, 0.0, 0.0])

        filtered = retrieval._semantic_pruning(candidates, query_dir)

        assert len(filtered) == 0


class TestGravitationalProjection:
    """测试引力投影"""

    def test_computes_states(self):
        """测试计算粒子状态"""
        retrieval = HMemRetrieval()

        candidates = [
            CandidateParticle(
                id="p1",
                direction=np.array([1.0, 0.0]),
                mass=1.0,
                temperature=1.0,
                initial_radius=2.0,
                created_at=0.0
            ),
        ]

        states = retrieval._gravitational_projection(candidates, t_now=0.0)

        assert len(states) == 1
        assert states[0].mass == 1.0
        assert states[0].hyperbolic_radius == 2.0
        assert states[0].memory_strength == 1.0

    def test_radius_decays_with_time(self):
        """测试半径随时间衰减"""
        retrieval = HMemRetrieval(config=RetrievalConfig(gamma=1.0))

        candidates = [
            CandidateParticle(
                id="p1",
                direction=np.array([1.0, 0.0]),
                mass=1.0,
                temperature=1.0,
                initial_radius=2.0,
                created_at=0.0
            ),
        ]

        states_t0 = retrieval._gravitational_projection(candidates, t_now=0.0)
        states_t1 = retrieval._gravitational_projection(candidates, t_now=1.0)

        # t=1 时的半径应该更小
        assert states_t1[0].hyperbolic_radius < states_t0[0].hyperbolic_radius


class TestThermodynamicScoring:
    """测试热力学采样"""

    def test_scores_higher_for_closer_particles(self):
        """测试距离更近的粒子分数更高"""
        retrieval = HMemRetrieval()

        # 创建查询状态
        query_state = retrieval.physics.compute_state(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=1.0,
            created_at=0.0,
            t_now=0.0
        )

        # 创建候选状态
        candidate_states = [
            retrieval.physics.compute_state(
                direction=np.array([1.0, 0.0]),  # 相同方向
                mass=1.0,
                temperature=1.0,
                initial_radius=1.0,
                created_at=0.0,
                t_now=0.0
            ),
            retrieval.physics.compute_state(
                direction=np.array([0.0, 1.0]),  # 不同方向
                mass=1.0,
                temperature=1.0,
                initial_radius=1.0,
                created_at=0.0,
                t_now=0.0
            ),
        ]

        results = retrieval._thermodynamic_scoring(candidate_states, query_state)

        assert len(results) == 2
        # 相同方向的粒子应该分数更高
        assert results[0].score > results[1].score

    def test_temperature_modulates_score(self):
        """测试温度调制分数"""
        retrieval = HMemRetrieval(config=RetrievalConfig(retrieval_beta=1.0))

        # 创建查询状态
        query_state = retrieval.physics.compute_state(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=1.0,
            created_at=0.0,
            t_now=0.0
        )

        # 创建两个相同位置但不同温度的候选状态
        high_T_state = retrieval.physics.compute_state(
            direction=np.array([0.9, 0.1]),  # 略微不同的方向
            mass=1.0,
            temperature=5.0,  # 高温（模糊记忆）
            initial_radius=1.0,
            created_at=0.0,
            t_now=0.0
        )

        low_T_state = retrieval.physics.compute_state(
            direction=np.array([0.9, 0.1]),  # 相同方向
            mass=1.0,
            temperature=0.5,  # 低温（清晰记忆）
            initial_radius=1.0,
            created_at=0.0,
            t_now=0.0
        )

        results = retrieval._thermodynamic_scoring(
            [high_T_state, low_T_state],
            query_state
        )

        # 由于方向不完全相同，高温（模糊）记忆应该获得更高的分数
        # 因为温度调制因子降低了距离惩罚
        assert results[0].temperature == 5.0


class TestInMemoryRetrieval:
    """测试内存检索系统"""

    def test_add_and_search(self):
        """测试添加和搜索"""
        retrieval = InMemoryRetrieval()
        t_now = 0.0  # 使用与创建时间相同的时间，避免衰减

        # 添加粒子
        retrieval.add_particle(create_candidate(
            particle_id="p1",
            direction=np.array([1.0, 0.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=1.0,
            created_at=t_now
        ))

        assert len(retrieval) == 1

        # 搜索
        results = retrieval.search(
            query_direction=np.array([1.0, 0.0, 0.0]),
            top_k=1,
            t_now=t_now  # 传递当前时间
        )

        assert len(results) == 1

    def test_search_returns_top_k(self):
        """测试返回 Top-K 结果"""
        retrieval = InMemoryRetrieval(
            config=RetrievalConfig(semantic_threshold=0.0)  # 不过滤
        )
        t_now = 0.0  # 使用固定时间

        # 添加多个粒子
        for i in range(10):
            angle = i * 0.1
            direction = np.array([np.cos(angle), np.sin(angle)])
            direction = direction / np.linalg.norm(direction)

            retrieval.add_particle(create_candidate(
                particle_id=f"p{i}",
                direction=direction,
                mass=1.0,
                temperature=1.0,
                initial_radius=1.0,
                created_at=t_now
            ))

        results = retrieval.search(
            query_direction=np.array([1.0, 0.0]),
            top_k=3,
            t_now=t_now  # 传递当前时间
        )

        assert len(results) <= 3

    def test_forgotten_particles_not_returned(self):
        """测试已遗忘的粒子不被返回"""
        retrieval = InMemoryRetrieval(
            config=RetrievalConfig(gamma=1.0, forgetting_threshold=0.1)
        )

        # 添加一个会快速遗忘的粒子
        retrieval.add_particle(create_candidate(
            particle_id="p1",
            direction=np.array([1.0, 0.0]),
            mass=0.1,  # 小质量，快速遗忘
            temperature=1.0,
            initial_radius=1.0,
            created_at=0.0
        ))

        # 搜索（很长时间后）
        results = retrieval.search(
            query_direction=np.array([1.0, 0.0]),
            t_now=100.0,  # 很长时间后
            top_k=10
        )

        # 粒子应该已被遗忘，不返回结果
        assert len(results) == 0


class TestCompleteWorkflow:
    """测试完整检索流程"""

    def test_three_step_retrieval(self):
        """测试三步检索流程"""
        retrieval = InMemoryRetrieval()

        # 创建一些候选粒子
        t_start = 0.0
        candidates = [
            # p1: 与查询相同方向，高质量
            create_candidate("p1", np.array([1.0, 0.0]), mass=2.0, temperature=0.5, initial_radius=2.0, created_at=t_start),
            # p2: 与查询相同方向，低质量
            create_candidate("p2", np.array([1.0, 0.0]), mass=0.5, temperature=0.5, initial_radius=0.5, created_at=t_start),
            # p3: 与查询垂直方向
            create_candidate("p3", np.array([0.0, 1.0]), mass=2.0, temperature=1.0, initial_radius=2.0, created_at=t_start),
        ]

        retrieval.add_particles(candidates)

        # 执行搜索
        results = retrieval.search(
            query_direction=np.array([1.0, 0.0]),
            query_mass=1.0,
            query_temperature=1.0,
            query_initial_radius=1.0,
            top_k=10,
            t_now=t_start
        )

        # p1 和 p2 应该被返回（方向相同），p3 不应该
        assert len(results) >= 1
        assert all("p3" not in r.id for r in results)

    def test_mass_affects_retrieval_order(self):
        """测试质量影响检索排序"""
        retrieval = InMemoryRetrieval(
            config=RetrievalConfig(semantic_threshold=0.9, gamma=1.0)
        )

        t_start = 0.0
        t_search = 5.0

        # 创建两个相同方向但不同质量的粒子
        candidates = [
            create_candidate("heavy", np.array([1.0, 0.0]), mass=5.0, temperature=1.0, initial_radius=2.0, created_at=t_start),
            create_candidate("light", np.array([1.0, 0.0]), mass=0.5, temperature=1.0, initial_radius=2.0, created_at=t_start),
        ]

        retrieval.add_particles(candidates)

        results = retrieval.search(
            query_direction=np.array([1.0, 0.0]),
            query_mass=1.0,
            query_temperature=1.0,
            query_initial_radius=1.0,
            top_k=10,
            t_now=t_search
        )

        # 质量大的粒子应该排在前面（因为衰减慢，保持更接近查询点）
        assert results[0].id == "heavy"


def run_tests():
    """运行所有测试"""
    print("Running Retrieval V3 Tests...")
    print("=" * 60)

    # 配置测试
    print("\n1. Testing RetrievalConfig...")
    test = TestRetrievalConfig()
    test.test_default_config()
    test.test_custom_config()
    print("   ✓ RetrievalConfig tests passed")

    # 语义过滤测试
    print("\n2. Testing semantic pruning...")
    test = TestSemanticPruning()
    test.test_filters_by_similarity()
    test.test_empty_when_no_match()
    print("   ✓ semantic pruning tests passed")

    # 引力投影测试
    print("\n3. Testing gravitational projection...")
    test = TestGravitationalProjection()
    test.test_computes_states()
    test.test_radius_decays_with_time()
    print("   ✓ gravitational projection tests passed")

    # 热力学采样测试
    print("\n4. Testing thermodynamic scoring...")
    test = TestThermodynamicScoring()
    test.test_scores_higher_for_closer_particles()
    test.test_temperature_modulates_score()
    print("   ✓ thermodynamic scoring tests passed")

    # 内存检索测试
    print("\n5. Testing InMemoryRetrieval...")
    test = TestInMemoryRetrieval()
    test.test_add_and_search()
    test.test_search_returns_top_k()
    test.test_forgotten_particles_not_returned()
    print("   ✓ InMemoryRetrieval tests passed")

    # 完整流程测试
    print("\n6. Testing complete workflow...")
    test = TestCompleteWorkflow()
    test.test_three_step_retrieval()
    test.test_mass_affects_retrieval_order()
    print("   ✓ complete workflow tests passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_tests()
