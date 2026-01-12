"""
粒子时间演化可视化验证

验证并可视化：
1. 速度衰减曲线（指数衰减）
2. 温度冷却曲线（模拟退火）
3. 距离变化曲线（精确积分）
4. 不同纯度粒子的对比
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/mnt/d/Codes/HyperAmy')

from particle.particle import Particle
from poincare.physics import ParticleProjector, TimePhysics
import time

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_test_particle(entity_id: str, emotion_vector: np.ndarray,
                        purity: float, T_min: float = 0.1, T_max: float = 1.0,
                        alpha: float = 0.5, tau_base: float = 86400.0,
                        beta: float = 1.0, gamma: float = 2.0) -> dict:
    """
    创建测试粒子（返回字典格式，包含所有物理属性）

    Args:
        entity_id: 实体ID
        emotion_vector: 情绪向量
        purity: 归一化纯度 [0, 1]

    Returns:
        包含粒子所有属性的字典
    """
    t_born = time.time()

    # 计算权重（模长）
    weight = float(np.linalg.norm(emotion_vector))

    # 归一化向量
    if weight > 1e-9:
        normalized_vector = emotion_vector / weight
    else:
        normalized_vector = emotion_vector.copy()
        weight = 0.0

    # 计算初始温度（基于纯度）
    T0 = T_min + (T_max - T_min) * (1.0 - purity)

    # 计算初始速度（基于模长和纯度）
    v0 = weight * (1.0 + alpha * purity)

    # 计算时间常数（基于纯度）
    tau_v = tau_base * (1.0 + gamma * purity)
    tau_T = tau_base * (1.0 + beta * purity)

    return {
        'entity_id': entity_id,
        'emotion_vector': normalized_vector,
        'weight': weight,
        'speed': v0,
        'temperature': T0,
        'purity': purity,
        'tau_v': tau_v,
        'tau_T': tau_T,
        'born': t_born
    }


def simulate_evolution(particle: dict, time_points: np.ndarray, T_min: float = 0.1):
    """
    模拟粒子在不同时间点的状态

    Args:
        particle: 粒子字典
        time_points: 时间点数组（秒）
        T_min: 最小温度

    Returns:
        包含时间演化数据的字典
    """
    results = {
        'time_hours': time_points / 3600,  # 转换为小时
        'speed': [],
        'temperature': [],
        'distance_integrated': [],  # 精确积分距离
        'v_current': [],
        'T_current': []
    }

    v0 = particle['speed']
    T0 = particle['temperature']
    tau_v = particle['tau_v']
    tau_T = particle['tau_T']
    born = particle['born']
    weight = particle['weight']

    for t in time_points:
        t_now = born + t

        # 计算当前速度和温度
        v_current = TimePhysics.f(v0, born, t_now, tau_v)
        T_current = TimePhysics.g(T0, born, t_now, tau_T, T_min)

        # 计算累积距离（精确积分）
        # distance = v0 * tau_v * (1 - exp(-t/tau_v))
        integrated_distance = v0 * tau_v * (1.0 - np.exp(-t / tau_v))

        # 当前总距离 = 初始距离 + 累积距离
        # 假设 scaling_factor = 2.0
        scaling_factor = 2.0
        initial_distance = weight * scaling_factor
        total_distance = initial_distance + integrated_distance

        results['speed'].append(v0)  # 初始速度（参考）
        results['v_current'].append(v_current)
        results['temperature'].append(T0)  # 初始温度（参考）
        results['T_current'].append(T_current)
        results['distance_integrated'].append(total_distance)

    return results


def plot_evolution_curves():
    """
    绘制粒子时间演化曲线
    """
    print("\n" + "=" * 80)
    print("粒子时间演化可视化验证")
    print("=" * 80)

    # Create test particles with different purity levels
    test_cases = [
        {
            'name': 'High Purity (Stable)',
            'purity': 0.9,
            'color': 'green',
            'linestyle': '-'
        },
        {
            'name': 'Medium Purity (Average)',
            'purity': 0.5,
            'color': 'blue',
            'linestyle': '--'
        },
        {
            'name': 'Low Purity (Chaotic)',
            'purity': 0.1,
            'color': 'red',
            'linestyle': ':'
        }
    ]

    # 时间点：0 到 7天，每2小时一个点
    time_points = np.arange(0, 7 * 24 * 3600, 2 * 3600)  # 0-7天，步长2小时

    # 模拟每个粒子
    particles_data = []
    for case in test_cases:
        vec = np.random.randn(768)  # 随机情绪向量
        particle = create_test_particle(
            entity_id=f"test_purity_{case['purity']}",
            emotion_vector=vec,
            purity=case['purity']
        )
        results = simulate_evolution(particle, time_points)
        particles_data.append({
            'case': case,
            'particle': particle,
            'results': results
        })

        print(f"\n✓ {case['name']} (purity={case['purity']}):")
        print(f"  - 初始速度 v0 = {particle['speed']:.4f}")
        print(f"  - 初始温度 T0 = {particle['temperature']:.4f}")
        print(f"  - tau_v = {particle['tau_v']:.0f} 秒 = {particle['tau_v']/86400:.2f} 天")
        print(f"  - tau_T = {particle['tau_T']:.0f} 秒 = {particle['tau_T']/86400:.2f} 天")

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Particle Time Evolution Verification (Free Energy Principle)', fontsize=16, fontweight='bold')

    # Plot 1: Speed decay curve
    ax1 = axes[0, 0]
    for data in particles_data:
        case = data['case']
        results = data['results']
        ax1.plot(results['time_hours'], results['v_current'],
                label=case['name'], color=case['color'], linestyle=case['linestyle'],
                linewidth=2)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Current Speed v(t)', fontsize=12)
    ax1.set_title('Speed Decay Curve (Exponential Decay)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 7*24)

    # Plot 2: Temperature cooling curve
    ax2 = axes[0, 1]
    for data in particles_data:
        case = data['case']
        results = data['results']
        ax2.plot(results['time_hours'], results['T_current'],
                label=case['name'], color=case['color'], linestyle=case['linestyle'],
                linewidth=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Current Temperature T(t)', fontsize=12)
    ax2.set_title('Temperature Cooling Curve (Simulated Annealing)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 7*24)

    # Plot 3: Distance change curve (exact integration)
    ax3 = axes[1, 0]
    for data in particles_data:
        case = data['case']
        results = data['results']
        ax3.plot(results['time_hours'], results['distance_integrated'],
                label=case['name'], color=case['color'], linestyle=case['linestyle'],
                linewidth=2)
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_ylabel('Distance from Origin', fontsize=12)
    ax3.set_title('Distance Change Curve (Exact Integration v(t)dt)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 7*24)

    # Plot 4: Speed retention rate after 1 day
    ax4 = axes[1, 1]
    names = []
    retention_rates = []
    colors = []
    for data in particles_data:
        case = data['case']
        particle = data['particle']
        v0 = particle['speed']
        results = data['results']

        # Find speed after 24 hours
        idx_24h = np.argmin(np.abs(results['time_hours'] - 24))
        v_24h = results['v_current'][idx_24h]
        retention = (v_24h / v0) * 100

        names.append(case['name'])
        retention_rates.append(retention)
        colors.append(case['color'])

    bars = ax4.bar(names, retention_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Speed Retention Rate (%)', fontsize=12)
    ax4.set_title('Speed Retention Rate After 1 Day', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(True, axis='y', alpha=0.3)

    # Annotate values on bars
    for i, (bar, rate) in enumerate(zip(bars, retention_rates)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    output_path = Path('/mnt/d/Codes/HyperAmy/test/time_evolution_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存到: {output_path}")

    # 打印关键数据
    print("\n" + "=" * 80)
    print("关键数据验证（1天后）")
    print("=" * 80)

    for data in particles_data:
        case = data['case']
        particle = data['particle']
        results = data['results']

        v0 = particle['speed']
        T0 = particle['temperature']

        idx_24h = np.argmin(np.abs(results['time_hours'] - 24))
        v_24h = results['v_current'][idx_24h]
        T_24h = results['T_current'][idx_24h]
        dist_24h = results['distance_integrated'][idx_24h]

        print(f"\n【{case['name']}】")
        print(f"  初始状态:")
        print(f"    - v0 = {v0:.4f}")
        print(f"    - T0 = {T0:.4f}")
        print(f"    - purity = {particle['purity']:.4f}")
        print(f"    - tau_v = {particle['tau_v']/86400:.2f} 天")
        print(f"  1天后状态:")
        print(f"    - v(24h) = {v_24h:.4f} (保留 {v_24h/v0*100:.1f}%)")
        print(f"    - T(24h) = {T_24h:.4f} (下降 {T0-T_24h:.4f})")
        print(f"    - distance = {dist_24h:.2f}")

    print("\n" + "=" * 80)
    print("✅ 验证完成！")
    print("=" * 80)


def plot_exact_vs_approximate():
    """
    对比精确积分 vs 线性近似的误差
    """
    print("\n" + "=" * 80)
    print("Exact Integration vs Linear Approximation Comparison")
    print("=" * 80)

    # Parameters
    v0 = 1.0
    tau_v = 86400.0  # 1 day

    # Time points: 1 hour to 7 days
    time_hours = np.array([1, 6, 12, 24, 48, 72, 168])
    time_seconds = time_hours * 3600

    # Calculate both methods
    exact_distances = []
    approx_distances = []
    errors = []

    for dt in time_seconds:
        # Exact integration
        d_exact = v0 * tau_v * (1.0 - np.exp(-dt / tau_v))

        # Linear approximation
        v_current = v0 * np.exp(-dt / tau_v)
        d_approx = v_current * dt

        # Relative error
        error = abs(d_approx - d_exact) / d_exact * 100

        exact_distances.append(d_exact)
        approx_distances.append(d_approx)
        errors.append(error)

        print(f"  {dt/3600:6.1f} hours: exact={d_exact:10.2f}, approx={d_approx:10.2f}, error={error:6.2f}%")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: distance comparison
    ax1.plot(time_hours, exact_distances, 'g-o', label='Exact Integration', linewidth=2, markersize=8)
    ax1.plot(time_hours, approx_distances, 'r--s', label='Linear Approximation', linewidth=2, markersize=8)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Cumulative Distance', fontsize=12)
    ax1.set_title('Exact Integration vs Linear Approximation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Right plot: relative error
    bars = ax2.bar(time_hours, errors, color=['orange' if e < 10 else 'red' for e in errors],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title('Relative Error of Linear Approximation', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    # Add threshold line
    ax2.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='10% Threshold')
    ax2.legend(fontsize=11)

    # Annotate values on bars
    for i, (bar, err) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存图片
    output_path = Path('/mnt/d/Codes/HyperAmy/test/exact_vs_approximate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图已保存到: {output_path}")

    print("\n" + "=" * 80)
    print("✅ 验证完成！")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Particle Time Evolution Visualization Verification System")
    print("=" * 80)

    # Plot 1: Evolution curves
    plot_evolution_curves()

    # Plot 2: Exact integration vs linear approximation
    plot_exact_vs_approximate()

    print("\n" + "=" * 80)
    print("All Visualizations Complete!")
    print("=" * 80)
    print("\nGenerated charts:")
    print("  1. test/time_evolution_visualization.png - Evolution curves comparison")
    print("  2. test/exact_vs_approximate.png - Exact vs Approximate integration")
    print("=" * 80)
