# ExpandNRC

**ExpandNRC** is a Python package for computing emotion similarity and frequencies using an extended NRC Emotion Lexicon. It leverages transformer-based embeddings to generalize emotion associations and efficiently computes emotion distributions for text inputs.

---

## 📦 Features

- **Emotion Similarity Calculation**  
  Compute calibrated cosine similarity scores between input words and emotion-labeled lexicon entries using pre-trained transformers.

- **Caching**  
  Caches word embeddings to `~/.cache/ExpandNRC` to avoid recomputing them across sessions.

- **Threshold-Based Emotion Labeling**  
  Assign emotions based on a configurable similarity threshold, with fallback to `"neutral"`.

- **Text-Level Emotion Analysis**  
  Compute raw and normalized emotion frequencies over tokenized or raw input texts.

---

---

## ✨ Example Usage

```python
from ExpandNRC.emotion_distance_calculator import EmotionDistanceCalculator
from nrclex import NRCLex

feelings_nrc = NRCLex(
    "path/to/NRCLex/nrc_v3.json")
emotion_lexicon = feelings_nrc.__lexicon__

# Initialize distance-based calculator
calculator = EmotionDistanceCalculator(emotion_lexicon, device="cpu")

# Get emotion scores for a single word
print("Emotions for 'happy':", calculator.get_emotions("happy"))

# Get scores for multiple words
print("Batch result:", calculator.get_emotions(["happy", "sad", "morning"]))

# Get threshold-based NRC-style labels
print("NRC emotions:", calculator.nrc_emotions(["happy", "sad", "morning"]))
```

---

## 📊 Emotion Frequency Analysis

```python
from ExpandNRC.emotion_frequences import EmotionFrequencyCalculator
from nrclex import NRCLex

feelings_nrc = NRCLex(
    "path/to/NRCLex/nrc_v3.json")
emotion_lexicon = feelings_nrc.__lexicon__

# Use raw input
freq_calc = EmotionFrequencyCalculator(emotion_lexicon, input="I'm happy but also worried.")
print(freq_calc.affect_frequencies)
print(freq_calc.top_emotions)

# Use tokenized input
tokens = ["happy", "excited", "trust", "worried", "fear", "joy"]
freq_calc = EmotionFrequencyCalculator(emotion_lexicon)
freq_calc.load_token_list(tokens)
print(freq_calc.affect_frequencies)
```

---

## 🧠 Cache Management

Lexicon embeddings are cached automatically to avoid redundant computation.  
Cache path: `~/.cache/ExpandNRC`

- If you update your lexicon, the cache will automatically refresh.
- You can delete the cache manually to force full re-embedding.

---

## 📄 License

MIT License