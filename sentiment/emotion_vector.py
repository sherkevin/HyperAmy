"""
Emotion Vector Extraction Module

Extracts emotion vectors from text using LLMs
"""
import numpy as np
from typing import Optional
from point_label.emotion import Emotion

class EmotionExtractor:
    """
    Emotion vector extractor wrapper
    
    Provides compatibility with sentiment module interface
    """
    
    def __init__(self, model_name=None, enable_cache: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize EmotionExtractor
        
        Args:
            model_name: Model name to use, defaults to DEFAULT_MODEL
            enable_cache: Whether to enable emotion vector caching (default: True) - NOTE: currently not used
            cache_dir: Cache directory (if None, uses default path) - NOTE: currently not used
        """
        # Emotion class on server may not support enable_cache parameter yet
        # Try with enable_cache first, fallback to model_name only
        try:
            self.emotion = Emotion(model_name=model_name, enable_cache=enable_cache, cache_dir=cache_dir)
        except TypeError:
            # Fallback: Emotion class doesn't support these parameters yet
            self.emotion = Emotion(model_name=model_name)
    
    def extract_emotion_vector(self, text: str) -> np.ndarray:
        """
        Extract emotion vector from text
        
        Args:
            text: Input text
            
        Returns:
            numpy.ndarray: Normalized emotion vector
        """
        return self.emotion.extract(text)

