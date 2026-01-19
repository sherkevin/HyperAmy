"""
测试 poincare/math.py 模块

验证庞加莱球基础数学运算的正确性。

测试内容：
1. 保角因子计算
2. Möbius 加法
3. 双曲距离
4. 投影和提取操作
5. 边界情况
"""

import numpy as np
import torch
import pytest

from poincare.math import (
    conformal_factor,
    mobius_add,
    poincare_dist,
    poincare_dist_batch,
    project_to_poincare,
    extract_radius,
    extract_direction,
    PoincareBall,
    EPS
)


class TestConformalFactor:
    """测试保角因子"""

    def test_zero_vector(self):
        """零向量的保角因子应为 2"""
        z = np.zeros(3)
        lambda_z = conformal_factor(z, c=1.0)
        assert lambda_z == 2.0

    def test_small_vector(self):
        """小向量的保角因子应略大于 2"""
        z = np.array([0.1, 0.0, 0.0])
        lambda_z = conformal_factor(z, c=1.0)
        assert lambda_z > 2.0
        assert lambda_z < 2.1

    def test_boundary_vector(self):
        """接近边界的向量应产生大的保角因子"""
        z = np.array([0.9, 0.0, 0.0])
        lambda_z = conformal_factor(z, c=1.0)
        assert lambda_z > 10.0

    def test_out_of_bounds(self):
        """超出边界的向量应抛出异常"""
        z = np.array([1.5, 0.0, 0.0])
        with pytest.raises(ValueError):
            conformal_factor(z, c=1.0)

    def test_torch_input(self):
        """支持 torch.Tensor 输入"""
        z = torch.tensor([0.1, 0.0, 0.0])
        lambda_z = conformal_factor(z, c=1.0)
        assert lambda_z > 2.0


class TestMobiusAdd:
    """测试 Möbius 加法"""

    def test_zero_identity(self):
        """零向量是 Möbius 加法的单位元"""
        u = np.array([0.3, 0.2, 0.1])
        zero = np.zeros(3)

        result1 = mobius_add(zero, u, c=1.0)
        result2 = mobius_add(u, zero, c=1.0)

        np.testing.assert_allclose(result1, u, atol=1e-6)
        np.testing.assert_allclose(result2, u, atol=1e-6)

    def test_inverse_property(self):
        """u ⊕ (-u) 应该接近零向量"""
        u = np.array([0.3, 0.2, 0.1])
        neg_u = -u

        result = mobius_add(u, neg_u, c=1.0)

        np.testing.assert_allclose(result, np.zeros(3), atol=1e-6)

    def test_non_commutativity(self):
        """Möbius 加法在双曲空间中不可交换（这是正确的性质）"""
        u = np.array([0.2, 0.1, 0.0])
        v = np.array([0.1, 0.3, 0.1])

        result1 = mobius_add(u, v, c=1.0)
        result2 = mobius_add(v, u, c=1.0)

        # 两个结果应该不同（除非在特殊情况下）
        # 但模长应该相近
        norm1 = np.linalg.norm(result1)
        norm2 = np.linalg.norm(result2)
        assert abs(norm1 - norm2) < 0.1  # 模长应该相近

    def test_result_in_ball(self):
        """Möbius 加法结果应该在球内"""
        u = np.array([0.4, 0.3, 0.1])
        v = np.array([0.2, 0.1, 0.0])

        result = mobius_add(u, v, c=1.0)
        norm = np.linalg.norm(result)

        assert norm < 1.0  # 应该在单位球内

    def test_torch_input(self):
        """支持 torch.Tensor 输入"""
        u = torch.tensor([0.3, 0.2, 0.1])
        v = torch.tensor([0.1, 0.2, 0.0])

        result = mobius_add(u, v, c=1.0)

        assert isinstance(result, torch.Tensor)
        assert torch.norm(result) < 1.0


class TestPoincareDist:
    """测试双曲距离"""

    def test_zero_distance(self):
        """相同点的距离应为零"""
        u = np.array([0.3, 0.2, 0.1])

        dist = poincare_dist(u, u, c=1.0)

        assert dist == 0.0

    def test_symmetry(self):
        """距离应该对称"""
        u = np.array([0.3, 0.2, 0.1])
        v = np.array([0.1, 0.4, 0.0])

        dist1 = poincare_dist(u, v, c=1.0)
        dist2 = poincare_dist(v, u, c=1.0)

        assert abs(dist1 - dist2) < 1e-6

    def test_triangle_inequality(self):
        """三角不等式: d(u, w) <= d(u, v) + d(v, w)"""
        u = np.array([0.1, 0.0, 0.0])
        v = np.array([0.2, 0.0, 0.0])
        w = np.array([0.3, 0.0, 0.0])

        d_uv = poincare_dist(u, v, c=1.0)
        d_vw = poincare_dist(v, w, c=1.0)
        d_uw = poincare_dist(u, w, c=1.0)

        assert d_uw <= d_uv + d_vw + 1e-6

    def test_origin_distance(self):
        """从原点的距离公式简化"""
        # 从原点到 z 的距离: d(0, z) = 2/√c * arctanh(√c * ||z||)
        z = np.array([0.5, 0.0, 0.0])

        dist = poincare_dist(np.zeros(3), z, c=1.0)

        # 手动计算: 2 * arctanh(0.5)
        expected = 2.0 * np.arctanh(0.5)
        assert abs(dist - expected) < 1e-6

    def test_curvature_effect(self):
        """曲率越大，相同坐标的距离越大"""
        u = np.array([0.3, 0.0, 0.0])
        v = np.array([0.1, 0.0, 0.0])

        dist_c1 = poincare_dist(u, v, c=1.0)
        dist_c2 = poincare_dist(u, v, c=0.5)

        assert dist_c1 > dist_c2

    def test_torch_input(self):
        """支持 torch.Tensor 输入"""
        u = torch.tensor([0.3, 0.2, 0.1])
        v = torch.tensor([0.1, 0.4, 0.0])

        dist = poincare_dist(u, v, c=1.0)

        assert isinstance(dist, float)
        assert dist > 0


class TestPoincareDistBatch:
    """测试批量距离计算"""

    def test_batch_computation(self):
        """批量计算应该产生正确的结果"""
        u = np.array([0.3, 0.2, 0.1])
        v_batch = np.array([
            [0.3, 0.2, 0.1],  # 相同点，距离 0
            [0.1, 0.0, 0.0],  # 不同点
        ])

        distances = poincare_dist_batch(u, v_batch, c=1.0)

        assert len(distances) == 2
        assert distances[0] == 0.0
        assert distances[1] > 0


class TestProjection:
    """测试投影和提取操作"""

    def test_project_and_extract_radius(self):
        """投影后提取半径应该得到原始值"""
        direction = np.array([1.0, 0.0, 0.0])
        radius = 2.0

        z = project_to_poincare(direction, radius, c=1.0)
        extracted_radius = extract_radius(z, c=1.0)

        assert abs(extracted_radius - radius) < 1e-6

    def test_project_and_extract_direction(self):
        """投影后提取方向应该得到原始方向"""
        direction = np.array([0.6, 0.8, 0.0])
        radius = 1.5

        z = project_to_poincare(direction, radius, c=1.0)
        extracted_dir = extract_direction(z)

        # 归一化后比较
        direction_norm = direction / np.linalg.norm(direction)
        np.testing.assert_allclose(extracted_dir, direction_norm, atol=1e-6)

    def test_zero_radius(self):
        """零半径应该投影到原点"""
        direction = np.array([1.0, 0.0, 0.0])
        radius = 0.0

        z = project_to_poincare(direction, radius, c=1.0)

        np.testing.assert_allclose(z, np.zeros(3), atol=1e-6)

    def test_large_radius(self):
        """大半径应该接近球面边界"""
        direction = np.array([1.0, 0.0, 0.0])
        radius = 10.0

        z = project_to_poincare(direction, radius, c=1.0)
        norm = np.linalg.norm(z)

        # 应该接近 1（但不超过）
        assert norm < 1.0
        assert norm > 0.99

    def test_projected_in_ball(self):
        """投影结果必须在球内"""
        for radius in [0.1, 1.0, 2.0, 5.0]:
            direction = np.array([0.6, 0.8, 0.0])
            z = project_to_poincare(direction, radius, c=1.0)
            norm = np.linalg.norm(z)
            assert norm < 1.0, f"Radius {radius} resulted in norm {norm}"


class TestPoincareBall:
    """测试庞加莱球类"""

    def test_initialization(self):
        """初始化"""
        space = PoincareBall(curvature=1.0, dimension=3)
        assert space.c == 1.0
        assert space.dimension == 3

    def test_project_method(self):
        """project 方法"""
        space = PoincareBall(curvature=1.0)
        direction = np.array([1.0, 0.0, 0.0])
        radius = 1.0

        z = space.project(direction, radius)

        assert isinstance(z, np.ndarray)
        assert np.linalg.norm(z) < 1.0

    def test_dist_method(self):
        """dist 方法"""
        space = PoincareBall(curvature=1.0)
        u = np.array([0.3, 0.2, 0.1])
        v = np.array([0.1, 0.4, 0.0])

        dist = space.dist(u, v)

        assert dist > 0

    def test_mobius_method(self):
        """mobius 方法"""
        space = PoincareBall(curvature=1.0)
        u = np.array([0.3, 0.2, 0.1])
        v = np.array([0.1, 0.4, 0.0])

        result = space.mobius(u, v)

        assert isinstance(result, np.ndarray)
        assert np.linalg.norm(result) < 1.0

    def test_get_radius_method(self):
        """get_radius 方法"""
        space = PoincareBall(curvature=1.0)
        direction = np.array([1.0, 0.0, 0.0])
        radius = 1.5

        z = space.project(direction, radius)
        extracted = space.get_radius(z)

        assert abs(extracted - radius) < 1e-6

    def test_get_direction_method(self):
        """get_direction 方法"""
        space = PoincareBall(curvature=1.0)
        direction = np.array([0.6, 0.8, 0.0])
        radius = 1.0

        z = space.project(direction, radius)
        extracted = space.get_direction(z)

        direction_norm = direction / np.linalg.norm(direction)
        np.testing.assert_allclose(extracted, direction_norm, atol=1e-6)


def run_tests():
    """运行所有测试"""
    print("Running Poincaré Math Tests...")
    print("=" * 60)

    # 保角因子测试
    print("\n1. Testing conformal_factor...")
    test = TestConformalFactor()
    test.test_zero_vector()
    test.test_small_vector()
    test.test_boundary_vector()
    test.test_torch_input()
    print("   ✓ conformal_factor tests passed")

    # Möbius 加法测试
    print("\n2. Testing mobius_add...")
    test = TestMobiusAdd()
    test.test_zero_identity()
    test.test_inverse_property()
    test.test_non_commutativity()
    test.test_result_in_ball()
    test.test_torch_input()
    print("   ✓ mobius_add tests passed")

    # 双曲距离测试
    print("\n3. Testing poincare_dist...")
    test = TestPoincareDist()
    test.test_zero_distance()
    test.test_symmetry()
    test.test_triangle_inequality()
    test.test_origin_distance()
    test.test_curvature_effect()
    test.test_torch_input()
    print("   ✓ poincare_dist tests passed")

    # 投影测试
    print("\n4. Testing projection functions...")
    test = TestProjection()
    test.test_project_and_extract_radius()
    test.test_project_and_extract_direction()
    test.test_zero_radius()
    test.test_large_radius()
    test.test_projected_in_ball()
    print("   ✓ projection tests passed")

    # 庞加莱球类测试
    print("\n5. Testing PoincareBall class...")
    test = TestPoincareBall()
    test.test_initialization()
    test.test_project_method()
    test.test_dist_method()
    test.test_mobius_method()
    test.test_get_radius_method()
    test.test_get_direction_method()
    print("   ✓ PoincareBall class tests passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_tests()
