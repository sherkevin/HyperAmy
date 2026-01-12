"""
Thermodynamics module: Free energy-based quantities

Implements:
- Energy: E(e) = -log p(e|context)
- Temperature: T ∝ Var(E) ∝ (1-P)/P
- Free energy: F = E - TS
"""
from typing import Dict, Optional
import numpy as np
from hipporag.utils.logging_utils import get_logger
from .purity import Purity

logger = get_logger(__name__)


class Thermodynamics:
    """
    Thermodynamic quantities computation based on free energy principle
    """

    def __init__(
        self,
        T_min: float = 0.1,
        T_max: float = 1.0,
        eps: float = 1e-10
    ):
        """
        Initialize Thermodynamics calculator

        Args:
            T_min: Minimum temperature (zero temperature, ordered state)
            T_max: Maximum temperature (high temperature, disordered state)
            eps: Numerical stability constant
        """
        self.T_min = T_min
        self.T_max = T_max
        self.eps = eps
        self.purity_calculator = Purity(eps=eps)

        logger.info(
            f"Thermodynamics module initialized "
            f"(T_min={T_min}, T_max={T_max})"
        )

    def compute_energy(
        self,
        emotion_vector: np.ndarray,
        context_mean: Optional[np.ndarray] = None,
        context_cov: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute energy E(e) = -log p(e|context)

        Under Gaussian assumption:
        E(e) = 0.5 * (e - μ)ᵀ Σ⁻¹ (e - μ) + const

        Simplified: E = 0.5 * ||e||² (assuming μ=0, Σ=I)

        Args:
            emotion_vector: Emotion vector
            context_mean: Context mean vector (optional, default = zeros)
            context_cov: Context covariance matrix (optional, default = I)

        Returns:
            Energy value
        """
        e = emotion_vector.astype(np.float64)

        if context_mean is None:
            context_mean = np.zeros_like(e)

        if context_cov is None:
            # Simplified: E = 0.5 * ||e||²
            energy = 0.5 * np.sum((e - context_mean) ** 2)
        else:
            # Full Gaussian energy
            diff = e - context_mean
            precision = np.linalg.inv(context_cov + self.eps * np.eye(len(e)))
            energy = 0.5 * diff @ precision @ diff

        return float(energy)

    def compute_temperature(
        self,
        emotion_vector: np.ndarray
    ) -> float:
        """
        Compute temperature based on purity

        Formula: T = T_min + (T_max - T_min) * (1 - normalized_purity)

        Physical meaning:
        - High purity → Low temperature (ordered, stable)
        - Low purity → High temperature (disordered, unstable)

        Args:
            emotion_vector: Emotion vector

        Returns:
            Temperature in [T_min, T_max]
        """
        # Compute normalized purity
        purity_norm = self.purity_calculator.compute_normalized(emotion_vector)

        # Temperature = inverse of purity
        # High purity → Low T, Low purity → High T
        temperature = self.T_min + (self.T_max - self.T_min) * (1.0 - purity_norm)

        return float(np.clip(temperature, self.T_min, self.T_max))

    def compute_free_energy(
        self,
        emotion_vector: np.ndarray,
        context_mean: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute free energy F = E - TS

        Args:
            emotion_vector: Emotion vector
            context_mean: Context mean (optional)

        Returns:
            Dictionary with energy, temperature, entropy, free_energy
        """
        # Energy
        energy = self.compute_energy(emotion_vector, context_mean)

        # Temperature
        temperature = self.compute_temperature(emotion_vector)

        # Entropy (Shannon entropy)
        entropy = self.purity_calculator.compute_entropy(emotion_vector)

        # Free energy
        free_energy = energy - temperature * entropy

        return {
            'energy': energy,
            'temperature': temperature,
            'entropy': entropy,
            'free_energy': free_energy
        }

    def compute_time_constants(
        self,
        emotion_vector: np.ndarray,
        tau_base: float = 86400.0,
        beta: float = 1.0,
        gamma: float = 2.0
    ) -> Dict[str, float]:
        """
        Compute time constants for evolution

        Args:
            emotion_vector: Emotion vector
            tau_base: Base time constant (seconds)
            beta: Temperature cooling coefficient
            gamma: Intensity decay coefficient

        Returns:
            Dictionary with tau_T (temperature), tau_v (intensity)
        """
        purity_norm = self.purity_calculator.compute_normalized(emotion_vector)

        # Time constants increase with purity
        tau_T = tau_base * (1.0 + beta * purity_norm)
        tau_v = tau_base * (1.0 + gamma * purity_norm)

        return {
            'tau_T': float(tau_T),
            'tau_v': float(tau_v)
        }

    def compute_batch_temperature(
        self,
        emotion_vectors: list,
        entity_ids: list,
        text_id: str
    ) -> list:
        """
        Compute temperature for multiple emotion vectors

        Args:
            emotion_vectors: List of emotion vectors
            entity_ids: Entity IDs (for logging)
            text_id: Text ID (for logging)

        Returns:
            List of temperature values
        """
        temperatures = []

        for i, (vec, eid) in enumerate(zip(emotion_vectors, entity_ids)):
            temperature = self.compute_temperature(vec)
            temperatures.append(temperature)

            logger.debug(
                f"Temperature computed: entity_id={eid}, "
                f"temperature={temperature:.4f}"
            )

        logger.info(
            f"Computed temperatures for {len(temperatures)} entities "
            f"(text_id={text_id})"
        )

        return temperatures
