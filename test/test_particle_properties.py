"""
测试 particle/properties.py 模块

验证粒子属性计算的正确性。

测试内容：
1. 情绪强度 I 计算
2. 分布纯度 κ 计算
3. 引力质量 m 计算
4. 热力学温度 T 计算
5. 初始双曲半径 R₀ 计算
6. 完整属性计算流程
7. 边界情况
"""

import numpy as np
import pytest

from particle.properties import (
    ParticleProperties,
    PropertyCalculator,
    compute_properties,
    get_default_calculator,
    DEFAULT_ALPHA_MASS,
    DEFAULT_BETA_MASS,
    DEFAULT_T0,
    MIN_KAPPA,
)


class TestIntensity:
    """测试情绪强度计算"""

    def test_unit_vector(self):
        """单位向量的强度应为 1"""
        calc = PropertyCalculator()
        vec = np.array([1.0, 0.0, 0.0])
        intensity = calc.compute_intensity(vec)
        assert intensity == 1.0

    def test_scaled_vector(self):
        """缩放向量的强度应等于模长"""
        calc = PropertyCalculator()
        vec = np.array([3.0, 4.0, 0.0])
        intensity = calc.compute_intensity(vec)
        assert intensity == 5.0  # sqrt(3² + 4²) = 5

    def test_zero_vector(self):
        """零向量的强度应为 0"""
        calc = PropertyCalculator()
        vec = np.array([0.0, 0.0, 0.0])
        intensity = calc.compute_intensity(vec)
        assert intensity == 0.0

    def test_negative_values(self):
        """负值不应影响强度（模长）"""
        calc = PropertyCalculator()
        vec = np.array([-3.0, -4.0, 0.0])
        intensity = calc.compute_intensity(vec)
        assert intensity == 5.0


class TestPurity:
    """测试分布纯度计算"""

    def test_max_purity(self):
        """单峰分布应获得最大纯度 1"""
        calc = PropertyCalculator()
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        purity = calc.compute_purity(vec)
        assert purity == 1.0

    def test_min_purity(self):
        """均匀分布应获得最小纯度 1/d"""
        calc = PropertyCalculator()
        d = 5
        vec = np.ones(d) / d
        purity = calc.compute_purity(vec)
        expected = 1.0 / d
        assert abs(purity - expected) < 1e-6

    def test_medium_purity(self):
        """中等分布应获得中等纯度"""
        calc = PropertyCalculator()
        vec = np.array([0.7, 0.3, 0.0, 0.0])
        purity = calc.compute_purity(vec)
        assert 0.5 < purity < 1.0

    def test_zero_vector(self):
        """零向量应返回最小纯度"""
        calc = PropertyCalculator()
        vec = np.array([0.0, 0.0, 0.0])
        purity = calc.compute_purity(vec)
        assert purity == MIN_KAPPA

    def test_dimension_5(self):
        """5 维向量的最小纯度应为 0.2"""
        calc = PropertyCalculator()
        vec = np.ones(5)
        purity = calc.compute_purity(vec)
        min_purity = 1.0 / 5
        assert abs(purity - min_purity) < 1e-6


class TestMass:
    """测试引力质量计算"""

    def test_mass_formula(self):
        """验证质量公式 m = α·I + β·log(1 + κ)"""
        calc = PropertyCalculator(
            alpha_mass=2.0,
            beta_mass=3.0
        )
        intensity = 5.0
        purity = 1.0

        mass = calc.compute_mass(intensity, purity)

        # m = 2*5 + 3*log(2) = 10 + 3*0.693 = 10 + 2.079 = 12.079
        expected = 2.0 * 5.0 + 3.0 * np.log(2.0)
        assert abs(mass - expected) < 1e-6

    def test_intensity_contribution(self):
        """强度应线性贡献质量"""
        calc = PropertyCalculator(alpha_mass=1.0, beta_mass=0.0)
        mass1 = calc.compute_mass(1.0, 1.0)
        mass2 = calc.compute_mass(5.0, 1.0)
        assert abs(mass2 - mass1 * 5.0) < 1e-6

    def test_purity_contribution(self):
        """纯度应对数贡献质量（边际递减）"""
        calc = PropertyCalculator(alpha_mass=0.0, beta_mass=1.0)
        mass1 = calc.compute_mass(0.0, 0.5)
        mass2 = calc.compute_mass(0.0, 1.0)
        assert mass2 > mass1  # 更高纯度 → 更大质量
        assert mass2 < mass1 * 2  # 但边际递减

    def test_minimum_mass(self):
        """质量应有一个最小值"""
        calc = PropertyCalculator()
        mass = calc.compute_mass(0.0, MIN_KAPPA)
        assert mass >= 0.01

    def test_combined_effect(self):
        """强度和纯度都应影响质量"""
        calc = PropertyCalculator()
        mass_low = calc.compute_mass(1.0, 0.5)
        mass_high = calc.compute_mass(5.0, 1.0)
        assert mass_high > mass_low


class TestTemperature:
    """测试热力学温度计算"""

    def test_temperature_formula(self):
        """验证温度公式 T = T₀/κ"""
        calc = PropertyCalculator(T0=10.0)
        purity = 0.5
        temp = calc.compute_temperature(purity)
        assert temp == 20.0  # 10 / 0.5 = 20

    def test_high_purity_low_temp(self):
        """高纯度 → 低温度"""
        calc = PropertyCalculator(T0=1.0)
        temp1 = calc.compute_temperature(1.0)
        temp2 = calc.compute_temperature(0.5)
        assert temp2 > temp1

    def test_low_purity_high_temp(self):
        """低纯度 → 高温度"""
        calc = PropertyCalculator(T0=1.0)
        temp = calc.compute_temperature(0.1)
        assert temp > 5.0

    def test_max_temperature_clipping(self):
        """温度应有上限"""
        calc = PropertyCalculator(T0=1.0, min_kappa=0.01)
        temp = calc.compute_temperature(0.001)  # 极低纯度
        assert temp == calc.T0 / calc.min_kappa  # 应被限制

    def test_temperature_range(self):
        """温度应在合理范围内"""
        calc = PropertyCalculator(T0=1.0)
        for purity in [0.2, 0.5, 0.8, 1.0]:
            temp = calc.compute_temperature(purity)
            assert temp >= 1.0  # T0
            assert temp <= 1.0 / MIN_KAPPA  # 最大值


class TestInitialRadius:
    """测试初始双曲半径计算"""

    def test_radius_proportional_to_mass(self):
        """半径应与质量成正比"""
        calc = PropertyCalculator(radius_scale=2.0)
        r1 = calc.compute_initial_radius(1.0)
        r2 = calc.compute_initial_radius(5.0)
        assert abs(r2 - r1 * 5.0) < 1e-6

    def test_scale_factor(self):
        """缩放系数应正确应用"""
        calc1 = PropertyCalculator(radius_scale=1.0)
        calc2 = PropertyCalculator(radius_scale=3.0)
        r1 = calc1.compute_initial_radius(2.0)
        r2 = calc2.compute_initial_radius(2.0)
        assert abs(r2 - r1 * 3.0) < 1e-6


class TestParticleProperties:
    """测试 ParticleProperties 数据类"""

    def test_dataclass_fields(self):
        """验证所有字段存在"""
        props = ParticleProperties(
            direction=np.array([1.0, 0.0]),
            intensity=1.0,
            purity=0.8,
            mass=1.5,
            temperature=1.25,
            initial_radius=1.5
        )
        assert props.intensity == 1.0
        assert props.purity == 0.8
        assert props.mass == 1.5
        assert props.temperature == 1.25
        assert props.initial_radius == 1.5

    def test_repr(self):
        """验证字符串表示"""
        props = ParticleProperties(
            direction=np.array([1.0, 0.0]),
            intensity=1.0,
            purity=0.8,
            mass=1.5,
            temperature=1.25,
            initial_radius=1.5
        )
        s = repr(props)
        assert "mass=1.5" in s
        assert "temperature=1.25" in s


class TestPropertyCalculator:
    """测试完整的属性计算流程"""

    def test_compute_properties(self):
        """测试完整的属性计算"""
        calc = PropertyCalculator(
            alpha_mass=1.0,
            beta_mass=1.0,
            T0=1.0,
            radius_scale=1.0
        )

        # 创建一个情绪向量：强度 5，中等纯度
        vec = np.array([3.0, 4.0, 0.0, 0.0])
        props = calc.compute_properties(vec)

        # 验证强度
        assert props.intensity == 5.0

        # 验证纯度（L2=5, L1=7, purity=25/49≈0.51）
        assert 0.4 < props.purity < 0.6

        # 验证质量
        assert props.mass > 0

        # 验证温度
        assert props.temperature >= 1.0

        # 验证方向（归一化）
        norm = np.linalg.norm(props.direction)
        assert abs(norm - 1.0) < 1e-6

    def test_pure_emotion(self):
        """纯情绪（单峰）应产生高纯度、低温度"""
        calc = PropertyCalculator(T0=1.0)
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        props = calc.compute_properties(vec)

        assert props.purity == 1.0
        assert props.temperature == 1.0  # T0 / 1 = T0

    def test_mixed_emotion(self):
        """混合情绪应产生较低纯度、较高温度"""
        calc = PropertyCalculator(T0=1.0)
        vec = np.array([0.25, 0.25, 0.25, 0.25])
        props = calc.compute_properties(vec)

        assert props.purity < 1.0
        assert props.temperature > 1.0

    def test_strong_emotion(self):
        """强情绪（高模长）应产生大质量"""
        calc = PropertyCalculator()
        weak = np.array([0.1, 0.1, 0.1])
        strong = np.array([5.0, 5.0, 0.0])

        props_weak = calc.compute_properties(weak)
        props_strong = calc.compute_properties(strong)

        assert props_strong.mass > props_weak.mass

    def test_zero_vector(self):
        """零向量应产生合理的默认值"""
        calc = PropertyCalculator()
        vec = np.array([0.0, 0.0, 0.0])
        props = calc.compute_properties(vec)

        assert props.intensity == 0.0
        assert props.mass >= 0.01  # 最小质量


class TestBatchComputation:
    """测试批量计算"""

    def test_compute_batch(self):
        """测试批量计算"""
        calc = PropertyCalculator()
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        result = calc.compute_batch(vectors)

        assert len(result['directions']) == 3
        assert len(result['intensities']) == 3
        assert len(result['purities']) == 3
        assert len(result['masses']) == 3
        assert len(result['temperatures']) == 3
        assert len(result['initial_radii']) == 3

    def test_compute_batch_with_ids(self):
        """测试带 ID 的批量计算"""
        calc = PropertyCalculator()
        vectors = [np.array([1.0, 0.0]), np.array([0.5, 0.5])]
        ids = ["entity1", "entity2"]

        result = calc.compute_batch(vectors, entity_ids=ids)

        assert len(result['directions']) == 2


class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_compute_properties_function(self):
        """测试便捷函数"""
        vec = np.array([3.0, 4.0, 0.0])
        props = compute_properties(vec)

        assert props.intensity == 5.0
        assert isinstance(props, ParticleProperties)

    def test_get_default_calculator(self):
        """测试默认计算器单例"""
        calc1 = get_default_calculator()
        calc2 = get_default_calculator()

        assert calc1 is calc2  # 应该是同一个实例


def run_tests():
    """运行所有测试"""
    print("Running Particle Properties Tests...")
    print("=" * 60)

    # 情绪强度测试
    print("\n1. Testing intensity computation...")
    test = TestIntensity()
    test.test_unit_vector()
    test.test_scaled_vector()
    test.test_zero_vector()
    test.test_negative_values()
    print("   ✓ intensity tests passed")

    # 纯度测试
    print("\n2. Testing purity computation...")
    test = TestPurity()
    test.test_max_purity()
    test.test_min_purity()
    test.test_medium_purity()
    test.test_zero_vector()
    test.test_dimension_5()
    print("   ✓ purity tests passed")

    # 质量测试
    print("\n3. Testing mass computation...")
    test = TestMass()
    test.test_mass_formula()
    test.test_intensity_contribution()
    test.test_purity_contribution()
    test.test_minimum_mass()
    test.test_combined_effect()
    print("   ✓ mass tests passed")

    # 温度测试
    print("\n4. Testing temperature computation...")
    test = TestTemperature()
    test.test_temperature_formula()
    test.test_high_purity_low_temp()
    test.test_low_purity_high_temp()
    test.test_max_temperature_clipping()
    test.test_temperature_range()
    print("   ✓ temperature tests passed")

    # 半径测试
    print("\n5. Testing initial radius computation...")
    test = TestInitialRadius()
    test.test_radius_proportional_to_mass()
    test.test_scale_factor()
    print("   ✓ initial radius tests passed")

    # 完整属性测试
    print("\n6. Testing full property computation...")
    test = TestPropertyCalculator()
    test.test_compute_properties()
    test.test_pure_emotion()
    test.test_mixed_emotion()
    test.test_strong_emotion()
    test.test_zero_vector()
    print("   ✓ full property computation tests passed")

    # 批量计算测试
    print("\n7. Testing batch computation...")
    test = TestBatchComputation()
    test.test_compute_batch()
    test.test_compute_batch_with_ids()
    print("   ✓ batch computation tests passed")

    # 便捷函数测试
    print("\n8. Testing convenience functions...")
    test = TestConvenienceFunctions()
    test.test_compute_properties_function()
    test.test_get_default_calculator()
    print("   ✓ convenience function tests passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_tests()
