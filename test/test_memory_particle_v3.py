"""
测试 particle/memory_particle.py 模块 (System V3)

验证记忆粒子的正确性。

测试内容：
1. 粒子创建
2. 双曲半径衰减
3. 庞加莱坐标更新
4. 引力时间膨胀效应（质量影响衰减）
5. 遗忘判断
6. 衰减曲线
"""

import numpy as np
import pytest
import math
import sys
import os

# 直接导入，避免 __init__.py 中的旧系统依赖问题
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from particle.memory_particle import (
    MemoryParticleV3,
    create_particle_from_emotion,
    create_particle_from_properties,
    ParticleUtils,
    DEFAULT_CURVATURE,
    DEFAULT_GAMMA,
)


class TestMemoryParticleV3:
    """测试记忆粒子类"""

    def test_initialization(self):
        """测试粒子初始化"""
        direction = np.array([1.0, 0.0, 0.0])
        mass = 2.0
        temperature = 1.0
        initial_radius = 3.0
        created_at = 100.0

        particle = MemoryParticleV3(
            direction=direction,
            mass=mass,
            temperature=temperature,
            initial_radius=initial_radius,
            created_at=created_at
        )

        assert particle.mass == mass
        assert particle.temperature == temperature
        assert particle.initial_radius == initial_radius
        assert particle.created_at == created_at
        assert particle.dimension == 3

    def test_direction_normalized(self):
        """方向向量应该自动归一化"""
        direction = np.array([3.0, 0.0, 0.0])

        particle = MemoryParticleV3(
            direction=direction,
            mass=1.0,
            temperature=1.0,
            initial_radius=1.0,
            created_at=0.0
        )

        norm = np.linalg.norm(particle.direction)
        assert abs(norm - 1.0) < 1e-6

    def test_age(self):
        """测试年龄计算"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=1.0,
            created_at=100.0
        )

        assert particle.age(110.0) == 10.0
        assert particle.age(100.0) == 0.0

    def test_hyperbolic_radius_initial(self):
        """初始双曲半径应该是 initial_radius"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=2.0,
            created_at=0.0
        )

        assert particle.hyperbolic_radius(0.0) == 2.0

    def test_hyperbolic_radius_decay(self):
        """双曲半径应该随时间衰减"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=2.0,
            created_at=0.0,
            gamma=1.0
        )

        # R(t) = R₀ * exp(-γ/m * t) = 2 * exp(-1 * t)
        t_start = 0.0
        R_0 = particle.hyperbolic_radius(t_start)
        R_1 = particle.hyperbolic_radius(t_start + 1.0)

        assert R_0 == 2.0
        assert abs(R_1 - 2.0 * math.exp(-1.0)) < 1e-6
        assert R_1 < R_0  # 衰减

    def test_mass_affects_decay(self):
        """质量应该影响衰减速率（质量在分母）"""
        # 低质量粒子
        light_particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=0.5,
            temperature=1.0,
            initial_radius=2.0,
            created_at=0.0,
            gamma=1.0
        )

        # 高质量粒子
        heavy_particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=2.0,
            temperature=1.0,
            initial_radius=2.0,
            created_at=0.0,
            gamma=1.0
        )

        # 经过相同时间
        t = 1.0
        light_radius = light_particle.hyperbolic_radius(t)
        heavy_radius = heavy_particle.hyperbolic_radius(t)

        # 质量小的衰减更快（γ/m 更大）
        assert light_radius < heavy_radius

    def test_poincare_coord(self):
        """测试庞加莱坐标计算"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=0.0,
            created_at=0.0,
            curvature=1.0
        )

        # 零半径应该投影到原点
        coord = particle.poincare_coord(0.0)
        np.testing.assert_allclose(coord, np.array([0.0, 0.0, 0.0]), atol=1e-6)

    def test_poincare_coord_in_ball(self):
        """庞加莱坐标应该在单位球内"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=10.0,  # 大半径
            created_at=0.0,
            curvature=1.0
        )

        coord = particle.poincare_coord(0.0)
        norm = np.linalg.norm(coord)

        # 应该在单位球内
        assert norm < 1.0

    def test_is_forgotten(self):
        """测试遗忘判断"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=1.0,
            created_at=0.0,
            gamma=1.0
        )

        # 初始时未遗忘
        assert not particle.is_forgotten(0.0, threshold=0.1)

        # 长时间后应该遗忘
        # R(t) = exp(-t), 当 R(t) < 0.1 时，t > ln(10) ≈ 2.3
        assert particle.is_forgotten(5.0, threshold=0.1)

    def test_euclidean_norm(self):
        """测试欧氏模长"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=0.0,
            created_at=0.0
        )

        # 零双曲半径 → 零欧氏模长
        assert particle.euclidean_norm(0.0) == 0.0

    def test_memory_strength(self):
        """测试记忆强度"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=2.0,
            created_at=0.0,
            gamma=1.0
        )

        # 初始强度为 1
        assert particle.memory_strength(0.0) == 1.0

        # 强度随时间衰减
        strength_1 = particle.memory_strength(1.0)
        assert 0 < strength_1 < 1.0

    def test_time_to_forget(self):
        """测试遗忘时间计算"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=1.0,
            created_at=0.0,
            gamma=1.0
        )

        # 手动计算遗忘时间
        # R(t) = exp(-t) = 0.001 → t = -ln(0.001) = ln(1000) ≈ 6.9
        # 需要传入 t_now=0.0，否则会使用当前系统时间
        time = particle.time_to_forget(threshold=0.001, t_now=0.0)
        expected = math.log(1.0 / 0.001)

        assert abs(time - expected) < 1e-6

    def test_repr(self):
        """测试字符串表示"""
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=2.5,
            temperature=1.2,
            initial_radius=3.0,
            created_at=0.0
        )

        s = repr(particle)
        assert "m=2.5" in s
        assert "T=1.2" in s


class TestCreateParticle:
    """测试粒子创建函数"""

    def test_create_from_emotion(self):
        """测试从情绪向量创建粒子"""
        emotion = np.array([3.0, 4.0, 0.0])
        particle = create_particle_from_emotion(emotion)

        assert particle.mass > 0
        assert particle.temperature > 0
        assert particle.initial_radius > 0
        assert particle.dimension == 3

    def test_create_with_parameters(self):
        """测试带参数创建粒子"""
        emotion = np.array([1.0, 0.0, 0.0])
        particle = create_particle_from_emotion(
            emotion,
            curvature=0.5,
            gamma=2.0,
            alpha_mass=0.5,
            beta_mass=0.5,
            T0=2.0,
            radius_scale=3.0
        )

        assert particle.curvature == 0.5
        assert particle.gamma == 2.0

    def test_create_from_properties(self):
        """测试从属性创建粒子"""
        direction = np.array([1.0, 0.0, 0.0])
        particle = create_particle_from_properties(
            direction=direction,
            mass=2.0,
            temperature=1.5,
            initial_radius=3.0
        )

        assert particle.mass == 2.0
        assert particle.temperature == 1.5
        assert particle.initial_radius == 3.0


class TestParticleUtils:
    """测试粒子工具类"""

    def test_decay_curve(self):
        """测试衰减曲线计算"""
        t_start = 0.0
        particle = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=1.0,
            temperature=1.0,
            initial_radius=2.0,
            created_at=t_start,
            gamma=1.0
        )

        time_points = [0.0, 1.0, 2.0, 3.0]
        curve = ParticleUtils.decay_curve(particle, time_points, t_start=t_start)

        assert len(curve['time']) == 4
        assert len(curve['radius']) == 4
        assert len(curve['strength']) == 4

        # 半径应该递减
        for i in range(1, len(curve['radius'])):
            assert curve['radius'][i] < curve['radius'][i - 1]

    def test_compare_decay(self):
        """测试多粒子衰减比较"""
        t_start = 0.0
        p1 = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=0.5,
            temperature=1.0,
            initial_radius=2.0,
            created_at=t_start,
            gamma=1.0
        )

        p2 = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=2.0,
            temperature=1.0,
            initial_radius=2.0,
            created_at=t_start,
            gamma=1.0
        )

        time_points = [0.0, 1.0, 2.0]
        results = ParticleUtils.compare_decay([p1, p2], time_points)

        assert 'particle_0' in results
        assert 'particle_1' in results

    def test_heavy_particle_decays_slower(self):
        """验证高质量粒子衰减更慢"""
        t_start = 0.0
        p_light = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=0.5,
            temperature=1.0,
            initial_radius=2.0,
            created_at=t_start,
            gamma=1.0
        )

        p_heavy = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=5.0,
            temperature=1.0,
            initial_radius=2.0,
            created_at=t_start,
            gamma=1.0
        )

        time_points = [0.0, 1.0, 2.0, 5.0, 10.0]
        results = ParticleUtils.compare_decay([p_light, p_heavy], time_points)

        # 在任何时间点（t=0 除外），轻质量粒子的半径都应该更小
        for i in range(1, len(time_points)):
            assert results['particle_0']['radius'][i] < results['particle_1']['radius'][i]


class TestTimeDilationEffect:
    """测试引力时间膨胀效应"""

    def test_trauma_memory(self):
        """创伤记忆（大质量）应该几乎不衰减"""
        # m → ∞ 时，衰减因子 → 1，R(t) ≈ R₀
        t_start = 0.0
        trauma = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=100.0,  # 大质量
            temperature=1.0,
            initial_radius=2.0,
            created_at=t_start,
            gamma=1.0
        )

        # 经过 10 秒
        R_10 = trauma.hyperbolic_radius(t_start + 10.0)

        # 衰减应该很小
        strength = R_10 / trauma.initial_radius
        assert strength > 0.9  # 仍然保持 90% 以上

    def test_trivia_memory(self):
        """琐碎记忆（小质量）应该快速遗忘"""
        # m → 0 时，衰减因子 → 0，R(t) → 0
        t_start = 0.0
        trivia = MemoryParticleV3(
            direction=np.array([1.0, 0.0]),
            mass=0.1,  # 小质量
            temperature=1.0,
            initial_radius=2.0,
            created_at=t_start,
            gamma=1.0
        )

        # 经过 5 秒
        R_5 = trivia.hyperbolic_radius(t_start + 5.0)

        # 衰减应该很大
        strength = R_5 / trivia.initial_radius
        assert strength < 0.1  # 只剩不到 10%


def run_tests():
    """运行所有测试"""
    print("Running Memory Particle V3 Tests...")
    print("=" * 60)

    # 基础测试
    print("\n1. Testing MemoryParticleV3 basic functionality...")
    test = TestMemoryParticleV3()
    test.test_initialization()
    test.test_direction_normalized()
    test.test_age()
    test.test_hyperbolic_radius_initial()
    test.test_hyperbolic_radius_decay()
    test.test_mass_affects_decay()
    test.test_poincare_coord()
    test.test_poincare_coord_in_ball()
    test.test_is_forgotten()
    test.test_euclidean_norm()
    test.test_memory_strength()
    test.test_time_to_forget()
    test.test_repr()
    print("   ✓ basic functionality tests passed")

    # 创建测试
    print("\n2. Testing particle creation...")
    test = TestCreateParticle()
    test.test_create_from_emotion()
    test.test_create_with_parameters()
    test.test_create_from_properties()
    print("   ✓ particle creation tests passed")

    # 工具测试
    print("\n3. Testing ParticleUtils...")
    test = TestParticleUtils()
    test.test_decay_curve()
    test.test_compare_decay()
    test.test_heavy_particle_decays_slower()
    print("   ✓ ParticleUtils tests passed")

    # 时间膨胀效应测试
    print("\n4. Testing gravitational time dilation effect...")
    test = TestTimeDilationEffect()
    test.test_trauma_memory()
    test.test_trivia_memory()
    print("   ✓ time dilation effect tests passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_tests()
