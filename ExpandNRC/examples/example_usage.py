"""
Example usage of the lexicon_expansion package.
"""

from nrclex import NRCLex

from ExpandNRC import EmotionDistanceCalculator

# Use local lexicon json for updated version of NRC
feelings_nrc = NRCLex("/Users/Panos/Library/CloudStorage/Dropbox/PI_Squared"
                      "/PycharmProjects/Research/NRCLex/nrc_v3.json")

emotion_lexicon = feelings_nrc.__lexicon__

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
