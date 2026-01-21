"""
Emotion Store Module

Manages persistent storage of emotion vectors using Parquet format
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EmotionStore:
    """
    Manages persistent storage of emotion vectors using Parquet format
    """
    
    def __init__(self, store_dir: str, namespace: str = "emotion", auto_save: bool = True, save_interval: int = 50):
        """
        Initialize EmotionStore
        
        Args:
            store_dir: Directory to store emotion vectors
            namespace: Namespace identifier for data segregation
            auto_save: Whether to auto-save after each set (default: True for backward compatibility)
            save_interval: Number of sets before auto-saving (only used if auto_save=True)
        """
        self.store_dir = store_dir
        self.namespace = namespace
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._pending_saves = 0  # Counter for pending saves
        
        if not os.path.exists(store_dir):
            logger.info(f"Creating emotion store directory: {store_dir}")
            os.makedirs(store_dir, exist_ok=True)
        
        self.filename = os.path.join(store_dir, f"emotion_{namespace}.parquet")
        self._load_data()
    
    def _load_data(self):
        """Load existing emotion vectors from parquet file"""
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.emotion_vectors = df["emotion_vector"].values.tolist()
            # Load mass if available (for backward compatibility)
            if "mass" in df.columns:
                self.masses = df["mass"].values.tolist()
            else:
                self.masses = [0.0] * len(self.hash_ids)
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            logger.info(f"Loaded {len(self.hash_ids)} emotion vectors from {self.filename}")
        else:
            self.hash_ids = []
            self.emotion_vectors = []
            self.masses = []
            self.hash_id_to_idx = {}
    
    def _save_data(self):
        """Save emotion vectors to parquet file"""
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "emotion_vector": self.emotion_vectors,
            "mass": self.masses if hasattr(self, 'masses') else [0.0] * len(self.hash_ids)
        })
        data_to_save.to_parquet(self.filename, index=False)
        logger.info(f"Saved {len(self.hash_ids)} emotion vectors to {self.filename}")
    
    def get(self, hash_id: str) -> Optional[np.ndarray]:
        """Get emotion vector by hash_id"""
        if hash_id in self.hash_id_to_idx:
            idx = self.hash_id_to_idx[hash_id]
            return np.array(self.emotion_vectors[idx])
        return None
    
    def set(self, hash_id: str, emotion_vector: np.ndarray, mass: float = 0.0, force_save: bool = False):
        """
        Store emotion vector and optional mass
        
        Args:
            hash_id: Hash ID of the document
            emotion_vector: Emotion vector to store
            mass: Mass value (default: 0.0)
            force_save: Force immediate save (default: False)
        """
        if hash_id in self.hash_id_to_idx:
            # Update existing
            idx = self.hash_id_to_idx[hash_id]
            self.emotion_vectors[idx] = emotion_vector.tolist()
            if hasattr(self, 'masses'):
                self.masses[idx] = mass
        else:
            # Add new
            self.hash_ids.append(hash_id)
            self.emotion_vectors.append(emotion_vector.tolist())
            if not hasattr(self, 'masses'):
                self.masses = [0.0] * len(self.hash_ids)
            self.masses.append(mass)
            self.hash_id_to_idx[hash_id] = len(self.hash_ids) - 1
        
        # Auto-save logic
        if self.auto_save:
            self._pending_saves += 1
            if force_save or self._pending_saves >= self.save_interval:
                self._save_data()
                self._pending_saves = 0
        elif force_save:
            self._save_data()
    
    def get_mass(self, hash_id: str) -> float:
        """Get mass value by hash_id"""
        if hash_id in self.hash_id_to_idx and hasattr(self, 'masses'):
            idx = self.hash_id_to_idx[hash_id]
            return self.masses[idx]
        return 0.0
    
    def batch_set(self, hash_ids: List[str], emotion_vectors: List[np.ndarray], masses: Optional[List[float]] = None, force_save: bool = True):
        """
        Batch store emotion vectors and optional masses
        
        Args:
            hash_ids: List of hash IDs
            emotion_vectors: List of emotion vectors
            masses: Optional list of mass values
            force_save: Whether to force save after batch (default: True)
        """
        if masses is None:
            masses = [0.0] * len(hash_ids)
        
        if not hasattr(self, 'masses'):
            self.masses = [0.0] * len(self.hash_ids)
        
        for hash_id, emotion_vector, mass in zip(hash_ids, emotion_vectors, masses):
            if hash_id in self.hash_id_to_idx:
                idx = self.hash_id_to_idx[hash_id]
                self.emotion_vectors[idx] = emotion_vector.tolist()
                self.masses[idx] = mass
            else:
                self.hash_ids.append(hash_id)
                self.emotion_vectors.append(emotion_vector.tolist())
                self.masses.append(mass)
                self.hash_id_to_idx[hash_id] = len(self.hash_ids) - 1
        
        if force_save:
            self._save_data()
            self._pending_saves = 0
        elif self.auto_save:
            self._pending_saves += len(hash_ids)
            if self._pending_saves >= self.save_interval:
                self._save_data()
                self._pending_saves = 0
    
    def flush(self):
        """Force save all pending changes"""
        if self._pending_saves > 0:
            self._save_data()
            self._pending_saves = 0
    
    def contains(self, hash_id: str) -> bool:
        """Check if emotion vector exists"""
        return hash_id in self.hash_id_to_idx
    
    def clear(self):
        """Clear all emotion vectors"""
        self.hash_ids = []
        self.emotion_vectors = []
        if hasattr(self, 'masses'):
            self.masses = []
        self.hash_id_to_idx = {}
        if os.path.exists(self.filename):
            os.remove(self.filename)
        logger.info("Cleared all emotion vectors")

