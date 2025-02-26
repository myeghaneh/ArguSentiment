"""
Lexicon Expansion Package

Provides tools for emotion lexicon expansion and computing emotion similarities.
"""

from .emotion_distance_calculator import EmotionDistanceCalculator
from .utils import create_emotion_vectors, _compute_similarity