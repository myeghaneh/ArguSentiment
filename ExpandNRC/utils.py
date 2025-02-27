"""
Utility functions for lexicon expansion and emotion similarity calculations.
"""

import numpy as np
from scipy.spatial.distance import cosine

def _compute_similarity(emb1, emb2):
    """
    Safely compute cosine similarity between two embeddings.

    Parameters
    ----------
    emb1 : np.ndarray
        First embedding vector.
    emb2 : np.ndarray
        Second embedding vector.

    Returns
    -------
    float
        Cosine similarity value between emb1 and emb2.
    """
    try:
        # Ensure vectors are 1D
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        return 1 - cosine(emb1, emb2)
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0

def create_emotion_vectors(calculator):
    """
    Create binary emotion vectors for each word in the lexicon.

    Parameters
    ----------
    calculator : EmotionDistanceCalculator
        Instance of the emotion distance calculator.

    Returns
    -------
    dict
        Mapping of words to their binary emotion vectors.
    """
    all_emotions = calculator.emotions
    emotion_vectors = {}

    for word, emotions in calculator.emotion_lexicon.items():
        vector = np.zeros(len(all_emotions))
        for emotion in emotions:
            if emotion in all_emotions:
                idx = all_emotions.index(emotion)
                vector[idx] = 1
        emotion_vectors[word] = vector

    return emotion_vectors