"""
Purity module: Compute purity of emotion vectors

Based on L2/L1 norm ratio, which measures directionality/concentration.
Purity represents the stability and certainty of emotion.
"""
from typing import List, Union
import numpy as np
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Purity:
    """Purity computation for emotion vectors"""

    def __init__(self, eps: float = 1e-10):
        """
        Initialize Purity calculator

        Args:
            eps: Small constant for numerical stability
        """
        self.eps = eps
        logger.info(f"Purity module initialized (eps={eps})")

    def compute(self, emotion_vector: np.ndarray) -> float:
        """
        Compute purity of a single emotion vector

        Formula: P(e) = ||e||_2² / ||e||_1²

        Args:
            emotion_vector: Emotion vector (not normalized)

        Returns:
            Purity value in [1/d, 1], where d is vector dimension
        """
        # L2 norm (Euclidean)
        l2_norm = np.linalg.norm(emotion_vector)

        # L1 norm (Manhattan)
        l1_norm = np.linalg.norm(emotion_vector, ord=1)

        # Avoid division by zero
        if l1_norm < self.eps:
            logger.warning("Zero L1 norm detected, returning minimum purity")
            return 1.0 / len(emotion_vector)

        # Purity = L2² / L1²
        purity = (l2_norm ** 2) / (l1_norm ** 2 + self.eps)

        # Clip to valid range [1/d, 1]
        min_purity = 1.0 / len(emotion_vector)
        purity = np.clip(purity, min_purity, 1.0)

        return float(purity)

    def compute_normalized(self, emotion_vector: np.ndarray) -> float:
        """
        Compute normalized purity in [0, 1]

        Formula: P_norm = (P - P_min) / (1 - P_min)

        Args:
            emotion_vector: Emotion vector

        Returns:
            Normalized purity in [0, 1]
        """
        purity = self.compute(emotion_vector)

        # Normalize to [0, 1]
        d = len(emotion_vector)
        min_purity = 1.0 / d
        max_purity = 1.0

        normalized = (purity - min_purity) / (max_purity - min_purity + self.eps)

        return float(np.clip(normalized, 0.0, 1.0))

    def compute_batch(self, emotion_vectors: List[np.ndarray]) -> List[float]:
        """
        Compute purity for multiple emotion vectors

        Args:
            emotion_vectors: List of emotion vectors

        Returns:
            List of purity values
        """
        purities = [self.compute(vec) for vec in emotion_vectors]
        logger.debug(f"Computed purities for {len(purities)} vectors")
        return purities

    def compute_entropy(self, emotion_vector: np.ndarray) -> float:
        """
        Compute Shannon entropy of the normalized emotion vector

        Formula: H(e) = -sum(p_i * log(p_i)), where p_i = |e_i| / ||e||_1

        Args:
            emotion_vector: Emotion vector

        Returns:
            Entropy in [0, log(d)]
        """
        # Normalize to probability distribution
        probs = np.abs(emotion_vector) / (np.linalg.norm(emotion_vector, ord=1) + self.eps)

        # Avoid log(0)
        probs = np.clip(probs, self.eps, 1.0)

        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs))

        return float(entropy)

    def compute_batch_normalized(self, emotion_vectors: List[np.ndarray]) -> List[float]:
        """
        Compute normalized purity for multiple emotion vectors

        Args:
            emotion_vectors: List of emotion vectors

        Returns:
            List of normalized purity values [0, 1]
        """
        purities_norm = [self.compute_normalized(vec) for vec in emotion_vectors]
        logger.debug(f"Computed normalized purities for {len(purities_norm)} vectors")
        return purities_norm
