"""
Gravitational Memory System (GMS) V2 - Memory Space Container

This module implements the memory space that manages all MemoryParticles.
It handles particle creation, time evolution, and dot-product retrieval.

Key Features:
- Dimension-agnostic: Accepts vectors of any dimension with consistency checking
- Configurable physics parameters: core_threshold, core_ratio, decay_alpha
- Global time stepping: Evolves all particles simultaneously
- Dot-product retrieval: Natural combination of semantic similarity and memory strength

Author: HyperAmy Team
Version: 2.0
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from .memory_particle import (
    MemoryParticle,
    create_particle,
    DEFAULT_CORE_THRESHOLD,
    DEFAULT_CORE_RATIO,
    DEFAULT_DECAY_ALPHA,
    DEFAULT_VANISH_THRESHOLD,
    ParticleUtils
)


@dataclass
class SpaceStatus:
    """Status report for the memory space."""
    total_particles: int
    alive_particles: int
    dead_particles: int
    core_memory_count: int
    average_norm: float
    average_core_norm: float
    dimension: int


class GravitationalMemorySpace:
    """
    Container for memory particles with Newtonian decay dynamics.

    The space manages all particles, handles time evolution, and provides
    dot-product based retrieval that naturally combines semantic similarity
    with memory strength (norm).

    Example:
        >>> space = GravitationalMemorySpace(
        ...     core_threshold=3.0,
        ...     core_ratio=0.8,
        ...     decay_alpha=0.1
        ... )
        >>> space.add(strong_memory_vector)  # norm ~ 10.0
        >>> space.add(weak_memory_vector)    # norm ~ 2.0
        >>> space.step(dt=1.0)  # Advance time by one turn
        >>> results = space.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        core_threshold: float = DEFAULT_CORE_THRESHOLD,
        core_ratio: float = DEFAULT_CORE_RATIO,
        decay_alpha: float = DEFAULT_DECAY_ALPHA,
        vanish_threshold: float = DEFAULT_VANISH_THRESHOLD,
        dimension: Optional[int] = None
    ):
        """
        Initialize the Gravitational Memory Space.

        Args:
            core_threshold: Norm threshold for core formation (default: 3.0)
            core_ratio: Ratio of excess norm converted to core (default: 0.8)
            decay_alpha: Newton cooling coefficient (default: 0.1)
            vanish_threshold: Norm below which particles are removed (default: 1e-3)
            dimension: Fixed dimension for all vectors (None = auto-detect from first vector)
        """
        self.core_threshold = core_threshold
        self.core_ratio = core_ratio
        self.decay_alpha = decay_alpha
        self.vanish_threshold = vanish_threshold
        self.dimension = dimension

        self.particles: List[MemoryParticle] = []
        self._total_added = 0
        self._total_removed = 0

    def add(self, vector: np.ndarray) -> MemoryParticle:
        """
        Add a new memory particle to the space.

        Args:
            vector: Input embedding vector (any dimension)

        Returns:
            The created MemoryParticle

        Raises:
            ValueError: If vector dimension doesn't match space dimension
        """
        vector = np.asarray(vector, dtype=np.float32)
        vec_dim = len(vector)

        # Dimension consistency check
        if self.dimension is None:
            self.dimension = vec_dim
        elif vec_dim != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {vec_dim}. "
                f"All vectors in the space must have the same dimension."
            )

        particle = create_particle(
            vector,
            core_threshold=self.core_threshold,
            core_ratio=self.core_ratio
        )

        self.particles.append(particle)
        self._total_added += 1

        return particle

    def step(self, dt: float = 1.0, auto_cleanup: bool = True) -> int:
        """
        Advance global time by dt units.

        All particles undergo Newtonian cooling decay. Dead particles (norm=0)
        are optionally removed to save memory.

        Args:
            dt: Time elapsed (1.0 = one interaction turn)
            auto_cleanup: If True, remove dead particles after step

        Returns:
            Number of particles removed (if auto_cleanup=True)
        """
        if not auto_cleanup:
            for p in self.particles:
                p.decay_step(dt, self.decay_alpha, self.vanish_threshold)
            return 0

        # Filter out dead particles
        alive_particles = []
        removed_count = 0

        for p in self.particles:
            is_alive = p.decay_step(dt, self.decay_alpha, self.vanish_threshold)
            if is_alive:
                alive_particles.append(p)
            else:
                removed_count += 1
                self._total_removed += 1

        self.particles = alive_particles
        return removed_count

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_score: float = -float('inf')
    ) -> List[Tuple[MemoryParticle, float]]:
        """
        Retrieve top-k memories using dot product scoring.

        Dot product = |query| * |memory| * cos(theta)
        This naturally combines:
        - Semantic similarity (cosine of angle)
        - Memory strength (norm of particle)

        Weak memories (low norm) will have low scores even with perfect direction match.

        Args:
            query_vector: Query embedding vector
            top_k: Maximum number of results to return
            min_score: Minimum score threshold (inclusive)

        Returns:
            List of (particle, score) tuples, sorted by score descending
        """
        if not self.particles:
            return []

        query_vector = np.asarray(query_vector, dtype=np.float32)

        # Check dimension
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )

        # Calculate dot products
        results = []
        for particle in self.particles:
            score = ParticleUtils.dot_product(query_vector, particle)
            if score >= min_score:
                results.append((particle, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def get_status(self) -> SpaceStatus:
        """
        Get current status of the memory space.

        Returns:
            SpaceStatus with current statistics
        """
        if not self.particles:
            return SpaceStatus(
                total_particles=0,
                alive_particles=0,
                dead_particles=0,
                core_memory_count=0,
                average_norm=0.0,
                average_core_norm=0.0,
                dimension=self.dimension or 0
            )

        alive = [p for p in self.particles if p.is_alive]
        core_memories = [p for p in alive if p.has_core]

        avg_norm = float(np.mean([p.current_norm for p in alive])) if alive else 0.0
        avg_core = float(np.mean([p.core_norm for p in core_memories])) if core_memories else 0.0

        return SpaceStatus(
            total_particles=len(self.particles),
            alive_particles=len(alive),
            dead_particles=len(self.particles) - len(alive),
            core_memory_count=len(core_memories),
            average_norm=avg_norm,
            average_core_norm=avg_core,
            dimension=self.dimension or 0
        )

    def get_particles_by_type(self) -> Dict[str, List[MemoryParticle]]:
        """
        Group particles by memory type.

        Returns:
            Dict with keys: 'core_memory', 'short_term', 'dead'
        """
        core_memory = []
        short_term = []
        dead = []

        for p in self.particles:
            if not p.is_alive:
                dead.append(p)
            elif p.has_core:
                core_memory.append(p)
            else:
                short_term.append(p)

        return {
            'core_memory': core_memory,
            'short_term': short_term,
            'dead': dead
        }

    def clear(self) -> None:
        """Remove all particles from the space."""
        self.particles.clear()

    def get_particle_vectors(self) -> np.ndarray:
        """
        Get all particle vectors as a matrix.

        Returns:
            Array of shape (n_particles, dimension)
        """
        if not self.particles:
            return np.array([]).reshape(0, self.dimension or 0)

        return np.vstack([p.raw_vector for p in self.particles])

    def get_particle_norms(self) -> np.ndarray:
        """
        Get all particle norms as a vector.

        Returns:
            Array of shape (n_particles,)
        """
        if not self.particles:
            return np.array([])
        return np.array([p.current_norm for p in self.particles])

    def simulate_decay(
        self,
        particle: MemoryParticle,
        time_steps: List[float]
    ) -> Dict[str, List[float]]:
        """
        Simulate the decay of a particle over time without modifying it.

        Useful for visualization and analysis. Time starts at 0 (simulated time).

        Args:
            particle: The particle to simulate
            time_steps: List of time values to simulate (starting from 0)

        Returns:
            Dict with 'time', 'norm', 'shell' lists
        """
        # Create a temporary copy for simulation with normalized time
        # Set created_at and last_updated to 0.0 for simulation
        sim_particle = MemoryParticle(
            unit_direction=particle.unit_direction.copy(),
            core_norm=particle.core_norm,
            current_norm=particle.current_norm,
            initial_norm=particle.initial_norm,
            created_at=0.0,
            last_updated=0.0,
            dimension=particle.dimension
        )

        times = []
        norms = []
        shells = []

        current_time = 0.0

        for t in time_steps:
            dt = t - current_time
            if dt > 0:
                sim_particle.decay_step(dt, self.decay_alpha, self.vanish_threshold)
                current_time = t

            times.append(t)
            norms.append(sim_particle.current_norm)
            shells.append(sim_particle.current_norm - sim_particle.core_norm)

        return {
            'time': times,
            'norm': norms,
            'shell': shells,
            'core': [particle.core_norm] * len(times)
        }

    def __len__(self) -> int:
        """Return number of particles in the space."""
        return len(self.particles)

    def __repr__(self) -> str:
        status = self.get_status()
        return (f"GravitationalMemorySpace(dim={status.dimension}, "
                f"particles={status.alive_particles}, "
                f"core_memories={status.core_memory_count}, "
                f"avg_norm={status.average_norm:.4f})")
