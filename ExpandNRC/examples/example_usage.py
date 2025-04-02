"""
Example usage of the lexicon_expansion package.
"""

from nrclex import NRCLex

from ExpandNRC.emotion_distance_calculator import EmotionDistanceCalculator

# Use local lexicon json for updated version of NRC
feelings_nrc = NRCLex("/NRCLex/nrc_v3.json")

emotion_lexicon = feelings_nrc.lexicon

# Initialize the calculator (using CPU for simplicity)
calculator = EmotionDistanceCalculator(emotion_lexicon, device="cpu")

# Get emotions for a single word
result_single = calculator.get_emotions("happy")
print("Emotions for 'happy':", result_single)

# Get emotions for a batch of words
result_batch = calculator.get_emotions(["happy", "sad", "morning"])
print("Emotions for batch:", result_batch)

nrc_emotions = calculator.nrc_emotions(["happy", "sad", "morning"])
print("NRC emotions:", nrc_emotions)

nrc_emotions_2 = calculator.nrc_emotions(["happy", "sad", "morning"],
                                         threshold=0.7)
print("NRC emotions after filtering:", nrc_emotions_2)
