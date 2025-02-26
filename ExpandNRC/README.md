
# ExpandNRC Package

The Lexicon Expansion Package provides tools to compute emotion similarities between words using a given emotion lexicon. Leveraging pre-trained transformer models, caching mechanisms, and calibrated similarity measures, the package allows you to quickly determine the emotions associated with words.

## Features

- **Emotion Similarity Calculation:**  
  Compute calibrated cosine similarity scores between word embeddings.

- **Caching:**  
  Cache lexicon embeddings in a user-specific directory (`~/.cache/ExpandNRC`) to avoid recalculating them on each run.

- **NRC Emotion Extraction:**  
  Retrieve emotion labels based on similarity threshold, with a fallback to `neutral` when no emotion exceeds the threshold.


# Example Usage

The following script demonstrates how to use the package with an updated NRC lexicon:

"""
Example usage of the ExpandNRC package.
"""
```python

from nrclex import NRCLex
from ExpandNRC import EmotionDistanceCalculator

# Load the local NRC lexicon JSON (update the path as necessary)
feelings_nrc = NRCLex("Path/to/NRCLex/nrc_v3.json")
emotion_lexicon = feelings_nrc.__lexicon__

# Initialize the calculator (using CPU for simplicity)
calculator = EmotionDistanceCalculator(emotion_lexicon, device="cpu")

# Get emotions with values for a single word
result_single = calculator.get_emotions("happy")
print("Emotions for 'happy':", result_single)

# Get emotions with values for a batch of words
result_batch = calculator.get_emotions(["happy", "sad", "morning"])
print("Emotions for batch:", result_batch)

# Get NRC emotions (emotion labels only)
nrc_emotions = calculator.nrc_emotions(["happy", "sad", "morning"])
print("NRC emotions:", nrc_emotions)
```

# Cache Management

The package automatically caches lexicon embeddings in ~/.cache/ExpandNRC to speed up subsequent runs. If the lexicon changes, the cache is recalculated.
