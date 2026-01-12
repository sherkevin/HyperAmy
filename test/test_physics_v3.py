"""
测试 poincare/physics.py 模块 (System V3)

验证物理引擎的正确性。

测试内容：
1. TimeDynamics: 时间膨胀衰变函数
2. FastPositionUpdate: 快速位置更新
3. PhysicsEngine: 统一接口
4. 质量对衰减的影响
"""

import numpy as np
import math

from poincare.physics import (
    ParticleState,
    TimeDynamics,
    FastPositionUpdate,
    PhysicsEngine,
    compute_particle_state,
    DEFAULT_GAMMA,
    FORGETTING_THRESHOLD,
)


class TestTimeDynamics:
    """测试时间动力学"""

    def test_hyperbolic_radius_initial(self):
        """初始时刻的双曲半径应该等于初始值"""
        R0 = 5.0
        mass = 2.0
        R_t = TimeDynamics.hyperbolic_radius(R0, mass, delta_t=0.0)
        assert R_t == R0

    def test_hyperbolic_radius_decay(self):
        """双曲半径应该随时间衰减"""
        R0 = 5.0
        mass = 1.0
        gamma = 1.0

        R_0 = TimeDynamics.hyperbolic_radius(R0, mass, 0.0, gamma)
        R_1 = TimeDynamics.hyperbolic_radius(R0, mass, 1.0, gamma)

        assert R_0 == R0
        assert R_1 < R_0  # 衰减
        # R(1) = 5 * exp(-1/1 * 1) = 5 * exp(-1)
        expected = 5.0 * math.exp(-1.0)
        assert abs(R_1 - expected) < 1e-6

    def test_mass_affects_decay(self):
        """质量应该影响衰减（质量在分母）"""
        R0 = 5.0
        gamma = 1.0

        R_light = TimeDynamics.hyperbolic_radius(R0, mass=0.5, delta_t=1.0, gamma=gamma)
        R_heavy = TimeDynamics.hyperbolic_radius(R0, mass=2.0, delta_t=1.0, gamma=gamma)

        # 质量小的衰减更快
        assert R_light < R_heavy

    def test_memory_strength(self):
        """记忆强度应该从 1 衰减到 0"""
        R0 = 5.0
        mass = 1.0
        gamma = 1.0

        s_0 = TimeDynamics.memory_strength(R0, mass, 0.0, gamma)
        s_1 = TimeDynamics.memory_strength(R0, mass, 1.0, gamma)

        assert s_0 == 1.0
        assert 0 < s_1 < 1.0

    def test_is_forgotten(self):
        """测试遗忘判断"""
        R0 = 1.0
        mass = 1.0
        gamma = 1.0
        threshold = 0.1

        # 初始时刻未遗忘
        assert not TimeDynamics.is_forgotten(R0, mass, 0.0, threshold, gamma)

        # 足够长时间后遗忘
        # R(t) = exp(-t) < 0.1 → t > ln(10) ≈ 2.3
        assert TimeDynamics.is_forgotten(R0, mass, 5.0, threshold, gamma)

    def test_time_to_forget(self):
        """测试遗忘时间计算"""
        R0 = 1.0
        mass = 1.0
        gamma = 1.0
        threshold = 0.001

        # t = (1/1) * ln(1/0.001) = ln(1000)
        time = TimeDynamics.time_to_forget(R0, mass, threshold, gamma)
        expected = math.log(1.0 / threshold)

        assert abs(time - expected) < 1e-6

    def test_zero_mass(self):
        """零质量应该返回零半径"""
        R0 = 5.0
        R_t = TimeDynamics.hyperbolic_radius(R0, mass=0.0, delta_t=1.0)
        assert R_t == 0.0


class TestFastPositionUpdate:
    """测试快速位置更新"""

    def test_poincare_coord_zero_radius(self):
        """零半径应该投影到原点"""
        direction = np.array([1.0, 0.0, 0.0])
        coord = FastPositionUpdate.poincare_coord(direction, hyperbolic_radius=0.0)
        np.testing.assert_allclose(coord, np.zeros(3), atol=1e-6)

    def test_poincare_coord_in_ball(self):
        """庞加莱坐标应该在单位球内"""
        direction = np.array([1.0, 0.0])
        coord = FastPositionUpdate.poincare_coord(direction, hyperbolic_radius=10.0)
        norm = np.linalg.norm(coord)
        assert norm < 1.0

    def test_compute_state(self):
        """测试完整状态计算"""
        direction = np.array([1.0, 0.0])
        mass = 2.0
        temp = 1.0
        R0 = 3.0
        created_at = 0.0
        t_now = 1.0

        state = FastPositionUpdate.compute_state(
            direction, mass, temp, R0, created_at, t_now
        )

        assert state.mass == mass
        assert state.temperature == temp
        assert state.hyperbolic_radius < R0  # 衰减
        assert 0 < state.memory_strength < 1.0
        assert not state.is_forgotten


class TestPhysicsEngine:
    """测试物理引擎"""

    def test_initialization(self):
        """测试初始化"""
        engine = PhysicsEngine(curvature=1.0, gamma=2.0, forgetting_threshold=0.01)
        assert engine.curvature == 1.0
        assert engine.gamma == 2.0
        assert engine.forgetting_threshold == 0.01

    def test_compute_state(self):
        """测试状态计算"""
        engine = PhysicsEngine()

        direction = np.array([1.0, 0.0])
        mass = 2.0
        temp = 1.0
        R0 = 3.0
        created_at = 0.0

        state = engine.compute_state(direction, mass, temp, R0, created_at, t_now=0.0)

        assert state.hyperbolic_radius == R0
        assert state.memory_strength == 1.0
        assert not state.is_forgotten

    def test_hyperbolic_radius_method(self):
        """测试 hyperbolic_radius 方法"""
        engine = PhysicsEngine(gamma=1.0)

        R0 = 5.0
        R_t = engine.hyperbolic_radius(R0, mass=1.0, delta_t=1.0)

        expected = 5.0 * math.exp(-1.0)
        assert abs(R_t - expected) < 1e-6

    def test_is_forgotten_method(self):
        """测试 is_forgotten 方法"""
        engine = PhysicsEngine(forgetting_threshold=0.1, gamma=1.0)

        # 初始时刻未遗忘
        assert not engine.is_forgotten(initial_radius=1.0, mass=1.0, delta_t=0.0)

        # 足够长时间后遗忘
        assert engine.is_forgotten(initial_radius=1.0, mass=1.0, delta_t=5.0)

    def test_time_to_forget_method(self):
        """测试 time_to_forget 方法"""
        engine = PhysicsEngine(gamma=1.0, forgetting_threshold=0.001)

        time = engine.time_to_forget(initial_radius=1.0, mass=1.0)
        expected = math.log(1.0 / 0.001)

        assert abs(time - expected) < 1e-6

    def test_compute_batch_states(self):
        """测试批量计算"""
        engine = PhysicsEngine()

        particles = [
            (np.array([1.0, 0.0]), 1.0, 1.0, 2.0, 0.0),
            (np.array([0.0, 1.0]), 2.0, 1.0, 3.0, 0.0),
        ]

        states = engine.compute_batch_states(particles, t_now=0.0)

        assert len(states) == 2
        assert states[0].mass == 1.0
        assert states[1].mass == 2.0


class TestConvenienceFunction:
    """测试便捷函数"""

    def test_compute_particle_state(self):
        """测试便捷函数"""
        direction = np.array([1.0, 0.0])
        state = compute_particle_state(
            direction=direction,
            mass=2.0,
            temperature=1.0,
            initial_radius=3.0,
            created_at=0.0,
            t_now=0.0
        )

        assert state.mass == 2.0
        assert state.hyperbolic_radius == 3.0


class TestPhysicsCorrectness:
    """测试物理正确性"""

    def test_trauma_memory_decays_slowly(self):
        """创伤记忆（大质量）应该衰减很慢"""
        R0 = 2.0
        gamma = 1.0
        dt = 10.0

        # 普通记忆
        R_normal = TimeDynamics.hyperbolic_radius(R0, mass=1.0, delta_t=dt, gamma=gamma)

        # 创伤记忆（大质量）
        R_trauma = TimeDynamics.hyperbolic_radius(R0, mass=100.0, delta_t=dt, gamma=gamma)

        # 创伤记忆的半径应该更大（衰减更慢）
        assert R_trauma > R_normal

        # 创伤记忆应该保持大部分强度
        strength_trauma = R_trauma / R0
        assert strength_trauma > 0.9

    def test_trivia_memory_decays_fast(self):
        """琐碎记忆（小质量）应该快速遗忘"""
        R0 = 2.0
        gamma = 1.0
        dt = 5.0

        # 琐碎记忆（小质量）
        R_trivia = TimeDynamics.hyperbolic_radius(R0, mass=0.1, delta_t=dt, gamma=gamma)

        # 琐碎记忆应该衰减到接近零
        strength_trivia = R_trivia / R0
        assert strength_trivia < 0.1


def run_tests():
    """运行所有测试"""
    print("Running Physics Engine V3 Tests...")
    print("=" * 60)

    # TimeDynamics 测试
    print("\n1. Testing TimeDynamics...")
    test = TestTimeDynamics()
    test.test_hyperbolic_radius_initial()
    test.test_hyperbolic_radius_decay()
    test.test_mass_affects_decay()
    test.test_memory_strength()
    test.test_is_forgotten()
    test.test_time_to_forget()
    test.test_zero_mass()
    print("   ✓ TimeDynamics tests passed")

    # FastPositionUpdate 测试
    print("\n2. Testing FastPositionUpdate...")
    test = TestFastPositionUpdate()
    test.test_poincare_coord_zero_radius()
    test.test_poincare_coord_in_ball()
    test.test_compute_state()
    print("   ✓ FastPositionUpdate tests passed")

    # PhysicsEngine 测试
    print("\n3. Testing PhysicsEngine...")
    test = TestPhysicsEngine()
    test.test_initialization()
    test.test_compute_state()
    test.test_hyperbolic_radius_method()
    test.test_is_forgotten_method()
    test.test_time_to_forget_method()
    test.test_compute_batch_states()
    print("   ✓ PhysicsEngine tests passed")

    # 便捷函数测试
    print("\n4. Testing convenience function...")
    test = TestConvenienceFunction()
    test.test_compute_particle_state()
    print("   ✓ convenience function tests passed")

    # 物理正确性测试
    print("\n5. Testing physics correctness...")
    test = TestPhysicsCorrectness()
    test.test_trauma_memory_decays_slowly()
    test.test_trivia_memory_decays_fast()
    print("   ✓ physics correctness tests passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_tests()
