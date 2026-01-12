"""
Gravitational Memory System V2 - Test Suite

Tests the 3 scenarios from system_v2.md:
1. Scenario A: The Trauma (Strong emotion event)
2. Scenario B: The Daily Trivia (Weak emotion event)
3. Scenario C: Specific vs Abstract (Resolution Check)

Uses 768-dimensional vectors for production environment simulation.
"""

import time
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from particle import (
    GravitationalMemorySpace,
    MemoryParticle,
    create_particle,
    SpaceStatus
)


# Test Configuration
DIMENSION = 768  # BERT base dimension
CORE_THRESHOLD = 3.0
CORE_RATIO = 0.8
DECAY_ALPHA = 0.1


def generate_random_vector(dimension: int, norm: float) -> np.ndarray:
    """Generate a random vector with specified norm."""
    vec = np.random.randn(dimension).astype(np.float32)
    current_norm = np.linalg.norm(vec)
    if current_norm > 0:
        vec = vec / current_norm * norm
    return vec


def scenario_a_trauma() -> Tuple[MemoryParticle, GravitationalMemorySpace, dict]:
    """
    Scenario A: The Trauma (Strong Emotion Event)

    Input: Vector with norm = 10.0

    Expectations:
    - Initial Core Norm should be ~8.0 (since 10 > 3.0 threshold)
    - After dt=10, norm should decay rapidly
    - After dt=1000, norm should stabilize at ~8.0 (core never vanishes)
    """
    print("=" * 80)
    print("SCENARIO A: The Trauma (Strong Emotion Event)")
    print("=" * 80)

    space = GravitationalMemorySpace(
        core_threshold=CORE_THRESHOLD,
        core_ratio=CORE_RATIO,
        decay_alpha=DECAY_ALPHA
    )

    # Create a strong memory vector (norm = 10.0)
    initial_norm = 10.0
    trauma_vector = generate_random_vector(DIMENSION, initial_norm)

    print(f"Input: Vector with norm = {initial_norm}")
    print(f"Core Threshold: {CORE_THRESHOLD}")
    print(f"Core Ratio: {CORE_RATIO}")

    particle = space.add(trauma_vector)

    print(f"\nInitial State:")
    print(f"  Initial Norm: {particle.initial_norm:.4f}")
    print(f"  Core Norm: {particle.core_norm:.4f}")
    print(f"  Expected Core: {(initial_norm - CORE_THRESHOLD) * CORE_RATIO + CORE_THRESHOLD:.4f}")
    print(f"  Has Core: {particle.has_core}")

    # Verify core formation
    expected_core = (initial_norm - CORE_THRESHOLD) * CORE_RATIO + CORE_THRESHOLD
    assert abs(particle.core_norm - expected_core) < 0.01, \
        f"Core norm mismatch: expected {expected_core}, got {particle.core_norm}"
    assert particle.has_core, "Strong memory should have a core"

    # Simulate decay over time
    time_points = [0, 1, 5, 10, 50, 100, 500, 1000]
    print(f"\nTime Evolution:")
    print(f"{'Time':>8} | {'Norm':>10} | {'Shell':>10} | {'Core':>10} | {'Alive':>6}")
    print("-" * 60)

    trajectory = {'time': [], 'norm': [], 'shell': [], 'core': []}

    for t in time_points:
        # Create a copy to simulate without modifying original
        sim_result = space.simulate_decay(particle, [t])
        trajectory['time'].extend(sim_result['time'])
        trajectory['norm'].extend(sim_result['norm'])
        trajectory['shell'].extend(sim_result['shell'])
        trajectory['core'].extend(sim_result['core'])

        norm = sim_result['norm'][-1]
        shell = sim_result['shell'][-1]
        core = sim_result['core'][-1]
        alive = norm > 0

        print(f"{t:>8} | {norm:>10.4f} | {shell:>10.4f} | {core:>10.4f} | {alive:>6}")

    # Verify long-term stability
    final_norm = trajectory['norm'][-1]
    assert abs(final_norm - particle.core_norm) < 0.5, \
        f"After long decay, norm should approach core. Expected ~{particle.core_norm}, got {final_norm}"

    print(f"\n✓ Scenario A PASSED: Trauma retains core memory after long decay")
    print(f"  Initial norm: {initial_norm}, Final norm: {final_norm:.4f}, Core: {particle.core_norm:.4f}")

    return particle, space, trajectory


def scenario_b_trivia() -> Tuple[MemoryParticle, GravitationalMemorySpace, dict]:
    """
    Scenario B: The Daily Trivia (Weak Emotion Event)

    Input: Vector with norm = 2.0

    Expectations:
    - Initial Core Norm should be 0.0 (below threshold 3.0)
    - After dt=10, norm should decay rapidly
    - After sufficient dt, norm should approach 0.0 (particle vanishes)
    """
    print("\n" + "=" * 80)
    print("SCENARIO B: The Daily Trivia (Weak Emotion Event)")
    print("=" * 80)

    space = GravitationalMemorySpace(
        core_threshold=CORE_THRESHOLD,
        core_ratio=CORE_RATIO,
        decay_alpha=DECAY_ALPHA
    )

    # Create a weak memory vector (norm = 2.0)
    initial_norm = 2.0
    trivia_vector = generate_random_vector(DIMENSION, initial_norm)

    print(f"Input: Vector with norm = {initial_norm}")
    print(f"Core Threshold: {CORE_THRESHOLD}")

    particle = space.add(trivia_vector)

    print(f"\nInitial State:")
    print(f"  Initial Norm: {particle.initial_norm:.4f}")
    print(f"  Core Norm: {particle.core_norm:.4f}")
    print(f"  Has Core: {particle.has_core}")

    # Verify no core formation
    assert particle.core_norm == 0.0, \
        f"Weak memory should have no core. Expected 0.0, got {particle.core_norm}"
    assert not particle.has_core, "Weak memory should not have a core"

    # Simulate decay over time
    time_points = [0, 1, 2, 5, 10, 20, 30, 50, 100]
    print(f"\nTime Evolution:")
    print(f"{'Time':>8} | {'Norm':>10} | {'Decay %':>10} | {'Alive':>6}")
    print("-" * 50)

    trajectory = {'time': [], 'norm': [], 'shell': [], 'core': []}

    for t in time_points:
        sim_result = space.simulate_decay(particle, [t])
        trajectory['time'].extend(sim_result['time'])
        trajectory['norm'].extend(sim_result['norm'])
        trajectory['shell'].extend(sim_result['shell'])
        trajectory['core'].extend(sim_result['core'])

        norm = sim_result['norm'][-1]
        decay_pct = (1 - norm / initial_norm) * 100
        alive = norm > 1e-3  # vanish threshold

        print(f"{t:>8} | {norm:>10.6f} | {decay_pct:>10.2f}% | {alive:>6}")

    # Verify eventual vanishing
    final_norm = trajectory['norm'][-1]
    assert final_norm < 0.1, \
        f"Weak memory should decay to near zero. Got {final_norm}"

    print(f"\n✓ Scenario B PASSED: Trivia memory decays to near zero")
    print(f"  Initial norm: {initial_norm}, Final norm: {final_norm:.6f}")

    return particle, space, trajectory


def scenario_c_resolution() -> Tuple[List[MemoryParticle], GravitationalMemorySpace, dict]:
    """
    Scenario C: Specific vs Abstract (Resolution Check)

    Input: Two semantically similar vectors (5-degree angle), both with norm = 10.0

    Expectations:
    - Initially: High Euclidean distance (easy to distinguish)
    - After decay: Euclidean distance decreases (vectors converge toward origin/core)
    - At low norm: Vectors become harder to distinguish (abstract/concept level)
    """
    print("\n" + "=" * 80)
    print("SCENARIO C: Specific vs Abstract (Resolution Check)")
    print("=" * 80)

    space = GravitationalMemorySpace(
        core_threshold=CORE_THRESHOLD,
        core_ratio=CORE_RATIO,
        decay_alpha=DECAY_ALPHA
    )

    # Create two similar vectors with ~5 degree angle
    # Use a base direction and add small perturbation
    base_direction = np.random.randn(DIMENSION).astype(np.float32)
    base_direction = base_direction / np.linalg.norm(base_direction)

    # Create perturbation (approximately 5 degrees)
    perturbation = np.random.randn(DIMENSION).astype(np.float32)
    # Make perturbation orthogonal to base
    perturbation = perturbation - np.dot(perturbation, base_direction) * base_direction
    perturbation = perturbation / np.linalg.norm(perturbation)

    # 5 degrees in radians
    angle = np.radians(5)
    tan_angle = np.tan(angle)

    # Create two vectors
    initial_norm = 10.0
    vec1 = (base_direction + tan_angle * perturbation * 0.5).astype(np.float32)
    vec1 = vec1 / np.linalg.norm(vec1) * initial_norm

    vec2 = (base_direction - tan_angle * perturbation * 0.5).astype(np.float32)
    vec2 = vec2 / np.linalg.norm(vec2) * initial_norm

    # Verify angle
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    actual_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    print(f"Input: Two vectors with norm = {initial_norm}")
    print(f"Angle between vectors: {actual_angle:.2f} degrees")

    particle1 = space.add(vec1)
    particle2 = space.add(vec2)

    print(f"\nInitial State:")
    print(f"  Particle 1 - Initial Norm: {particle1.initial_norm:.4f}, Core: {particle1.core_norm:.4f}")
    print(f"  Particle 2 - Initial Norm: {particle2.initial_norm:.4f}, Core: {particle2.core_norm:.4f}")

    # Calculate Euclidean distance over time
    time_points = [0, 1, 10, 50, 100]
    print(f"\nResolution Analysis:")
    print(f"{'Time':>8} | {'P1 Norm':>10} | {'P2 Norm':>10} | {'Euclidean Dist':>15} | {'Dot Product':>12}")
    print("-" * 80)

    results = {'time': [], 'p1_norm': [], 'p2_norm': [], 'euclidean_dist': [], 'dot_product': []}

    for t in time_points:
        sim_result1 = space.simulate_decay(particle1, [t])
        sim_result2 = space.simulate_decay(particle2, [t])

        p1_vec = particle1.unit_direction * sim_result1['norm'][-1]
        p2_vec = particle2.unit_direction * sim_result2['norm'][-1]

        euclidean_dist = np.linalg.norm(p1_vec - p2_vec)
        dot_prod = np.dot(p1_vec, p2_vec)

        results['time'].append(t)
        results['p1_norm'].append(sim_result1['norm'][-1])
        results['p2_norm'].append(sim_result2['norm'][-1])
        results['euclidean_dist'].append(euclidean_dist)
        results['dot_product'].append(dot_prod)

        print(f"{t:>8} | {results['p1_norm'][-1]:>10.4f} | {results['p2_norm'][-1]:>10.4f} | "
              f"{euclidean_dist:>15.6f} | {dot_prod:>12.4f}")

    # Verify that distance decreases
    initial_dist = results['euclidean_dist'][0]
    final_dist = results['euclidean_dist'][-1]

    print(f"\nDistance Analysis:")
    print(f"  Initial Euclidean Distance: {initial_dist:.6f}")
    print(f"  Final Euclidean Distance: {final_dist:.6f}")
    print(f"  Reduction: {(1 - final_dist/initial_dist) * 100:.2f}%")

    assert final_dist < initial_dist, \
        "Euclidean distance should decrease as particles decay"

    print(f"\n✓ Scenario C PASSED: Vectors converge as they decay (abstraction)")
    print(f"  Memories become more abstract (closer together) as specific details fade")

    return [particle1, particle2], space, results


def scenario_d_retrieval() -> None:
    """
    Scenario D: Dot Product Retrieval (Bonus)

    Verify that dot-product retrieval naturally incorporates memory strength.
    """
    print("\n" + "=" * 80)
    print("SCENARIO D: Dot Product Retrieval (Strength-Aware Ranking)")
    print("=" * 80)

    space = GravitationalMemorySpace(
        core_threshold=CORE_THRESHOLD,
        core_ratio=CORE_RATIO,
        decay_alpha=DECAY_ALPHA
    )

    # Create a base direction
    base_direction = np.random.randn(DIMENSION).astype(np.float32)
    base_direction = base_direction / np.linalg.norm(base_direction)

    # Create three memories with same direction but different strengths
    strong_vector = base_direction * 10.0  # Strong memory (will have core)
    medium_vector = base_direction * 5.0   # Medium memory
    weak_vector = base_direction * 2.0     # Weak memory (no core)

    p_strong = space.add(strong_vector)
    p_medium = space.add(medium_vector)
    p_weak = space.add(weak_vector)

    print(f"Created 3 memories with same direction, different norms:")
    print(f"  Strong: norm={p_strong.initial_norm:.2f}, core={p_strong.core_norm:.2f}")
    print(f"  Medium: norm={p_medium.initial_norm:.2f}, core={p_medium.core_norm:.2f}")
    print(f"  Weak:   norm={p_weak.initial_norm:.2f}, core={p_weak.core_norm:.2f}")

    # Query with same direction
    query = base_direction * 3.0

    print(f"\nQuery vector: norm=3.0, same direction as memories")

    results = space.search(query, top_k=10)

    print(f"\nRetrieval Results (sorted by dot product):")
    print(f"{'Rank':>6} | {'Type':>10} | {'Norm':>10} | {'Dot Product':>12}")
    print("-" * 50)

    for i, (particle, score) in enumerate(results, 1):
        mem_type = "Strong" if particle.has_core else "Weak"
        print(f"{i:>6} | {mem_type:>10} | {particle.current_norm:>10.4f} | {score:>12.4f}")

    # Verify ranking
    assert results[0][0] == p_strong, "Strong memory should rank first"
    assert results[1][0] == p_medium, "Medium memory should rank second"
    assert results[2][0] == p_weak, "Weak memory should rank last"

    # Now apply decay and check again
    print(f"\n--- After dt=50 (decay) ---")
    space.step(dt=50)

    results_after = space.search(query, top_k=10)

    print(f"Retrieval Results after decay:")
    print(f"{'Rank':>6} | {'Type':>10} | {'Norm':>10} | {'Dot Product':>12}")
    print("-" * 50)

    for i, (particle, score) in enumerate(results_after, 1):
        mem_type = "Strong" if particle.has_core else "Weak"
        print(f"{i:>6} | {mem_type:>10} | {particle.current_norm:>10.4f} | {score:>12.4f}")

    # Strong memory should still rank first (has core)
    assert results_after[0][0] == p_strong, "Strong memory should still rank first after decay"

    print(f"\n✓ Scenario D PASSED: Dot product naturally incorporates memory strength")


def visualize_results(
    trauma_traj: dict,
    trivia_traj: dict,
    save_path: str = "test/gms_v2_decay_curves.png"
) -> Figure:
    """Visualize the decay curves for Scenario A and B."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scenario A: Trauma
    ax1 = axes[0]
    ax1.plot(trauma_traj['time'], trauma_traj['norm'], 'b-', linewidth=2, label='Total Norm')
    ax1.plot(trauma_traj['time'], trauma_traj['core'], 'r--', linewidth=2, label='Core Norm')
    ax1.plot(trauma_traj['time'], trauma_traj['shell'], 'g:', linewidth=1.5, label='Shell Norm')
    ax1.axhline(y=8.0, color='r', linestyle=':', alpha=0.5, label='Expected Core (8.0)')
    ax1.set_xlabel('Time (interaction turns)')
    ax1.set_ylabel('Norm (Energy)')
    ax1.set_title('Scenario A: Trauma (Strong Memory)\nNorm = 10.0 → Core ≈ 8.0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 12)

    # Scenario B: Trivia
    ax2 = axes[1]
    ax2.plot(trivia_traj['time'], trivia_traj['norm'], 'b-', linewidth=2, label='Total Norm')
    ax2.axhline(y=0.001, color='r', linestyle=':', alpha=0.5, label='Vanish Threshold')
    ax2.set_xlabel('Time (interaction turns)')
    ax2.set_ylabel('Norm (Energy)')
    ax2.set_title('Scenario B: Trivia (Weak Memory)\nNorm = 2.0 → Vanishes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 2.5)

    plt.tight_layout()

    # Save figure
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Visualization saved to: {save_path}")

    return fig


def main():
    """Run all test scenarios."""
    print("\n" + "=" * 80)
    print("GRAVITATIONAL MEMORY SYSTEM V2 - TEST SUITE")
    print(f"Vector Dimension: {DIMENSION}")
    print("=" * 80)

    # Run scenarios
    _, _, trauma_traj = scenario_a_trauma()
    _, _, trivia_traj = scenario_b_trivia()
    _, _, resolution_traj = scenario_c_resolution()
    scenario_d_retrieval()

    # Visualize
    fig = visualize_results(trauma_traj, trivia_traj)
    plt.show()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Scenario A: Strong memories retain permanent core")
    print("  ✓ Scenario B: Weak memories decay to zero")
    print("  ✓ Scenario C: Memories converge (abstract) as they decay")
    print("  ✓ Scenario D: Dot product retrieval incorporates strength")


if __name__ == "__main__":
    main()
