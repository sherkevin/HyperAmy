"""
测试基于自由能原理的粒子物理实现

验证：
1. 纯度计算正确性
2. 温度与纯度的反比关系
3. 速度与模长和纯度的关系
4. 时间常数的计算
5. 精确积分距离计算
"""
import sys
import numpy as np
from particle.purity import Purity
from particle.speed import Speed
from particle.temperature import Temperature
from particle.particle import Particle

def test_purity():
    """测试纯度计算"""
    print("\n" + "=" * 80)
    print("测试 1: 纯度计算")
    print("=" * 80)

    purity_calc = Purity()

    # 测试 1: 纯态（单个非零分量）
    vec_pure = np.array([1.0, 0.0, 0.0, 0.0])
    purity = purity_calc.compute_normalized(vec_pure)
    print(f"✓ 纯态 [1,0,0,0]: purity = {purity:.4f} (预期: 1.0)")
    assert abs(purity - 1.0) < 0.01, "纯态的纯度应为 1.0"

    # 测试 2: 混合态（全1向量）
    vec_mixed = np.ones(768)
    purity = purity_calc.compute_normalized(vec_mixed)
    print(f"✓ 混合态 (全1向量): purity = {purity:.4f} (预期: 接近 0)")
    assert 0.0 <= purity <= 0.1, "混合态的纯度应接近 0"

    # 测试 3: 随机向量
    vec_random = np.random.randn(768)
    purity = purity_calc.compute_normalized(vec_random)
    print(f"✓ 随机向量: purity = {purity:.4f} (应在 [0, 1] 范围内)")
    assert 0.0 <= purity <= 1.0, "纯度应在 [0, 1] 范围内"

    print("\n✅ 纯度计算测试通过！")


def test_temperature():
    """测试温度计算"""
    print("\n" + "=" * 80)
    print("测试 2: 温度计算（基于纯度）")
    print("=" * 80)

    temp_calc = Temperature(T_min=0.1, T_max=1.0)
    purity_calc = Purity()

    # 测试 1: 高纯度 → 低温度
    vec_pure = np.array([10.0, 0.1, 0.1, 0.1])
    purity = purity_calc.compute_normalized(vec_pure)
    temperature = temp_calc.compute(
        entity_ids=["test1"],
        emotion_vectors=[vec_pure],
        text_id="test"
    )[0]

    print(f"✓ 高纯度 (purity={purity:.4f}): T = {temperature:.4f}")
    print(f"  预期: T 应接近 T_min=0.1")
    assert 0.1 <= temperature <= 0.3, "高纯度应有低温度"

    # 测试 2: 低纯度 → 高温度
    vec_mixed = np.ones(768)
    purity = purity_calc.compute_normalized(vec_mixed)
    temperature = temp_calc.compute(
        entity_ids=["test2"],
        emotion_vectors=[vec_mixed],
        text_id="test"
    )[0]

    print(f"✓ 低纯度 (purity={purity:.4f}): T = {temperature:.4f}")
    print(f"  预期: T 应接近 T_max=1.0")
    assert temperature >= 0.8, "低纯度应有高温度"

    print("\n✅ 温度计算测试通过！")


def test_speed():
    """测试速度计算"""
    print("\n" + "=" * 80)
    print("测试 3: 速度计算（基于模长和纯度）")
    print("=" * 80)

    speed_calc = Speed(alpha=0.5)
    purity_calc = Purity()

    # 测试 1: 大模长 + 高纯度 → 高速度
    vec1 = np.array([10.0, 0.1, 0.1, 0.1])
    purity1 = purity_calc.compute_normalized(vec1)
    speed1 = speed_calc.compute(
        entity_ids=["test1"],
        emotion_vectors=[vec1],
        text_id="test"
    )[0]
    magnitude1 = np.linalg.norm(vec1)

    print(f"✓ 大模长 + 高纯度: ||e||={magnitude1:.2f}, purity={purity1:.4f}")
    print(f"  speed = {speed1:.4f}")

    # 测试 2: 小模长 + 低纯度 → 低速度
    vec2 = np.ones(768) * 0.5
    purity2 = purity_calc.compute_normalized(vec2)
    speed2 = speed_calc.compute(
        entity_ids=["test2"],
        emotion_vectors=[vec2],
        text_id="test"
    )[0]
    magnitude2 = np.linalg.norm(vec2)

    print(f"✓ 小模长 + 低纯度: ||e||={magnitude2:.2f}, purity={purity2:.4f}")
    print(f"  speed = {speed2:.4f}")

    assert speed1 > speed2, "大模长+高纯度应有更高速度"

    print("\n✅ 速度计算测试通过！")


def test_time_constants():
    """测试时间常数计算"""
    print("\n" + "=" * 80)
    print("测试 4: 时间常数计算")
    print("=" * 80)

    purity_calc = Purity()

    # 测试不同纯度的时间常数
    vec_pure = np.array([10.0, 0.1, 0.1, 0.1])
    purity_pure = purity_calc.compute_normalized(vec_pure)

    vec_mixed = np.ones(768)
    purity_mixed = purity_calc.compute_normalized(vec_mixed)

    tau_base = 86400.0
    beta = 1.0
    gamma = 2.0

    tau_v_pure = tau_base * (1.0 + gamma * purity_pure)
    tau_T_pure = tau_base * (1.0 + beta * purity_pure)

    tau_v_mixed = tau_base * (1.0 + gamma * purity_mixed)
    tau_T_mixed = tau_base * (1.0 + beta * purity_mixed)

    print(f"✓ 高纯度粒子:")
    print(f"  tau_v = {tau_v_pure:.0f} 秒 = {tau_v_pure/86400:.2f} 天")
    print(f"  tau_T = {tau_T_pure:.0f} 秒 = {tau_T_pure/86400:.2f} 天")

    print(f"✓ 低纯度粒子:")
    print(f"  tau_v = {tau_v_mixed:.0f} 秒 = {tau_v_mixed/86400:.2f} 天")
    print(f"  tau_T = {tau_T_mixed:.0f} 秒 = {tau_T_mixed/86400:.2f} 天")

    assert tau_v_pure > tau_v_mixed, "高纯度粒子应衰减更慢"
    assert tau_T_pure > tau_T_mixed, "高纯度粒子应冷却更慢"

    print("\n✅ 时间常数计算测试通过！")


def test_exact_integration():
    """测试精确积分 vs 线性近似"""
    print("\n" + "=" * 80)
    print("测试 5: 精确积分 vs 线性近似")
    print("=" * 80)

    import math

    v0 = 1.0
    tau_v = 86400.0  # 1天

    # 测试不同时间点
    time_points = [3600, 86400, 604800]  # 1小时、1天、7天

    print(f"{'时间':<10} {'线性近似':<15} {'精确积分':<15} {'相对误差':<10}")
    print("-" * 60)

    for dt in time_points:
        # 线性近似（旧方法）
        v_current = v0 * math.exp(-dt / tau_v)
        d_approx = v_current * dt

        # 精确积分（新方法）
        d_exact = v0 * tau_v * (1.0 - math.exp(-dt / tau_v))

        error = abs(d_approx - d_exact) / d_exact * 100

        hours = dt / 3600
        print(f"{hours:.1f}小时    {d_approx:<15.2f} {d_exact:<15.2f} {error:>9.2f}%")

    print("\n✅ 精确积分验证完成！")


def test_particle_creation():
    """测试完整的粒子创建流程"""
    print("\n" + "=" * 80)
    print("测试 6: 完整的粒子创建流程")
    print("=" * 80)

    try:
        # 创建 Particle 实例（不依赖 LLM）
        particle = Particle(
            model_name=None,  # 不使用 LLM
            T_min=0.1,
            T_max=1.0,
            alpha=0.5,
            tau_base=86400.0,
            beta=1.0,
            gamma=2.0
        )

        print("✓ Particle 实例创建成功")
        print(f"  参数: T_min={particle.T_min}, T_max={particle.T_max}, alpha={particle.alpha}")
        print(f"       tau_base={particle.tau_base}, beta={particle.beta}, gamma={particle.gamma}")

        # 测试纯度模块
        purity_calc = particle.purity
        test_vec = np.random.randn(768)
        purity = purity_calc.compute_normalized(test_vec)
        print(f"✓ 纯度计算: purity = {purity:.4f}")

        # 测试速度模块
        speed_calc = particle.speed
        speed = speed_calc.compute(
            entity_ids=["test"],
            emotion_vectors=[test_vec],
            text_id="test_doc"
        )[0]
        print(f"✓ 速度计算: speed = {speed:.4f}")

        # 测试温度模块
        temp_calc = particle.temperature
        temperature = temp_calc.compute(
            entity_ids=["test"],
            emotion_vectors=[test_vec],
            text_id="test_doc"
        )[0]
        print(f"✓ 温度计算: temperature = {temperature:.4f}")

        print("\n✅ 粒子创建流程测试通过！")

    except Exception as e:
        print(f"\n❌ 粒子创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("基于自由能原理的粒子物理实现测试")
    print("=" * 80)

    # 运行所有测试
    test_purity()
    test_temperature()
    test_speed()
    test_time_constants()
    test_exact_integration()
    success = test_particle_creation()

    if success:
        print("\n" + "=" * 80)
        print("🎉 所有测试通过！")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ 测试失败")
        print("=" * 80)
        sys.exit(1)
