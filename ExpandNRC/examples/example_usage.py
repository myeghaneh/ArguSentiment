"""
Example usage of the lexicon_expansion package.
"""

from nrclex import NRCLex

from ExpandNRC.emotion_distance_calculator import EmotionDistanceCalculator

# # Use local lexicon json for updated version of NRC
feelings_nrc = NRCLex(
    "/Users/Panos/Library/CloudStorage/Dropbox/PI_Squared/PycharmProjects/Research"
    "/NRCLex/nrc_v3.json")
#
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

nrc_emotions_2 = calculator.nrc_emotions(["happy", "sad", "morning"],
                                         threshold=0.7)
print("NRC emotions after filtering:", nrc_emotions_2)


from ExpandNRC.emotion_frequences import EmotionFrequencyCalculator


# --- INPUT TEXT
text = "I feel hopeful and excited, but also a bit anxious and worried."

# --- OUR VERSION
our_calc = EmotionFrequencyCalculator(emotion_lexicon, text, threshold=0.6)
our_freq = our_calc.affect_frequencies
our_top = our_calc.top_emotions

# --- NRCLEX VERSION
feelings_nrc.load_raw_text(text)
nrc_freq = feelings_nrc.affect_frequencies
nrc_top = feelings_nrc.top_emotions
print(our_freq)

# --- PRINT COMPARISON
print("🔍 Emotion Frequency Comparison:\n")
print(f"{'Emotion':<15} {'Ours':>10} {'NRCLex':>10}")
print("-" * 37)

all_keys = sorted(set(our_freq.keys()) | set(nrc_freq.keys()))
for emotion in all_keys:
    print(
        f"{emotion:<15} {our_freq.get(emotion, 0):>10.2f} "
        f"{nrc_freq.get(emotion, 0):>10.2f}")

print("\n🏆 Top Emotions")
print("Ours:   ", our_top)
print("NRCLex: ", nrc_top)
