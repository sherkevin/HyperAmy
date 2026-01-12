"""
Temperature module: Compute temperature based on purity

Formula: T = T_min + (T_max - T_min) * (1 - purity)

Physical meaning:
- High purity → Low temperature (stable, ordered)
- Low purity → High temperature (unstable, disordered)
"""
from typing import List
import numpy as np
from hipporag.utils.logging_utils import get_logger
from .purity import Purity

logger = get_logger(__name__)


class Temperature:
    """
    Temperature computation based on purity

    Formula: T = T_min + (T_max - T_min) * (1 - purity)
    """

    def __init__(self, T_min: float = 0.1, T_max: float = 1.0):
        """
        Initialize Temperature calculator

        Args:
            T_min: Minimum temperature (ordered state)
            T_max: Maximum temperature (disordered state)
        """
        self.T_min = T_min
        self.T_max = T_max
        self.purity_calculator = Purity()
        logger.info(
            f"Temperature module initialized "
            f"(T_min={T_min}, T_max={T_max})"
        )

    def compute(
        self,
        entity_ids: List[str],
        emotion_vectors: List[np.ndarray],
        text_id: str
    ) -> List[float]:
        """
        Compute temperature for multiple emotion vectors

        Formula: T = T_min + (T_max - T_min) * (1 - purity_normalized)

        Args:
            entity_ids: Entity IDs (for logging)
            emotion_vectors: List of emotion vectors
            text_id: Text ID (for logging)

        Returns:
            List of temperature values
        """
        temperatures = []

        for i, (vec, eid) in enumerate(zip(emotion_vectors, entity_ids)):
            # Compute normalized purity
            purity_norm = self.purity_calculator.compute_normalized(vec)

            # Temperature = inverse of purity
            # High purity → Low T, Low purity → High T
            temperature = self.T_min + (self.T_max - self.T_min) * (1.0 - purity_norm)

            temperatures.append(temperature)

            logger.debug(
                f"Temperature computed: entity_id={eid}, "
                f"purity={purity_norm:.4f}, temperature={temperature:.4f}"
            )

        logger.info(
            f"Computed temperatures for {len(temperatures)} entities "
            f"(text_id={text_id})"
        )

        return temperatures
