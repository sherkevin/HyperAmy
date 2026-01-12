"""
Speed module: Compute initial intensity based on magnitude and purity

Formula: v = ||e|| * (1 + alpha * purity)

Physical meaning:
- Magnitude determines base intensity
- Purity provides enhancement (pure emotions are stronger)
"""
from typing import List
import numpy as np
from hipporag.utils.logging_utils import get_logger
from .purity import Purity

logger = get_logger(__name__)


class Speed:
    """
    Speed/Intensity computation based on magnitude and purity

    Formula: v = ||e|| * (1 + alpha * purity)
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize Speed calculator

        Args:
            alpha: Purity enhancement coefficient (default: 0.5)
        """
        self.alpha = alpha
        self.purity_calculator = Purity()
        logger.info(f"Speed module initialized (alpha={alpha})")

    def compute(
        self,
        entity_ids: List[str],
        emotion_vectors: List[np.ndarray],
        text_id: str
    ) -> List[float]:
        """
        Compute speed for multiple emotion vectors

        Formula: v = ||e|| * (1 + alpha * purity_normalized)

        Args:
            entity_ids: Entity IDs (for logging)
            emotion_vectors: List of emotion vectors
            text_id: Text ID (for logging)

        Returns:
            List of speed values
        """
        speeds = []

        for i, (vec, eid) in enumerate(zip(emotion_vectors, entity_ids)):
            # Compute magnitude
            magnitude = float(np.linalg.norm(vec))

            # Compute normalized purity
            purity_norm = self.purity_calculator.compute_normalized(vec)

            # Speed = magnitude * (1 + alpha * purity)
            speed = magnitude * (1.0 + self.alpha * purity_norm)

            speeds.append(speed)

            logger.debug(
                f"Speed computed: entity_id={eid}, "
                f"magnitude={magnitude:.4f}, purity={purity_norm:.4f}, "
                f"speed={speed:.4f}"
            )

        logger.info(f"Computed speeds for {len(speeds)} entities (text_id={text_id})")

        return speeds
