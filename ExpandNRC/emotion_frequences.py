# emotion_frequency_calculator.py

from collections import Counter
from textblob import TextBlob
from ExpandNRC.emotion_distance_calculator import EmotionDistanceCalculator


class EmotionFrequencyCalculator:
    """
    Computes emotion frequencies for a given input text or token list using a custom lexicon
    and EmotionDistanceCalculator.

    Parameters
    ----------
    lexicon : dict
        A dictionary mapping words to emotion labels.
    input : str or list of str, optional
        The raw text or list of tokens to analyze (default is None).
    is_tokenized : bool, optional
        Whether the input is already tokenized (default is False).
    threshold : float, optional
        The similarity threshold for assigning emotions (default is 0.5).
    """

    def __init__(self, lexicon, input=None, is_tokenized=False, threshold=0.5, device = "cpu"):
        self.text = ""
        self.words = []
        self.sentences = []
        self.affect_list = []
        self.affect_dict = {}
        self.raw_emotion_scores = {}
        self.affect_frequencies = {}
        self.top_emotions = []
        self.__lexicon__ = lexicon
        self.threshold = threshold

        self.distance_calculator = EmotionDistanceCalculator(lexicon, device = device)

        if input:
            if is_tokenized:
                self.load_token_list(input)
            else:
                self.load_raw_text(input)

    def load_token_list(self, token_list):
        """
        Load a pre-tokenized list of words.

        Parameters
        ----------
        token_list : list of str
            Tokenized input text to process.
        """
        self.text = ""
        self.words = token_list
        self.sentences = []
        self.__build_word_affect__()
        self.top_emotions = self._top_emotions()

    def load_raw_text(self, text):
        """
        Load and tokenize a raw text string using TextBlob.

        Parameters
        ----------
        text : str
            Raw input text to process.
        """
        self.text = text
        blob = TextBlob(text)
        self.words = [w.lemmatize() for w in blob.words]
        self.sentences = list(blob.sentences)
        self.__build_word_affect__()
        self.top_emotions = self._top_emotions()

    def __build_word_affect__(self):
        """
        Computes emotion distributions from the tokenized words using the emotion distance calculator.
        """
        affect_list = []
        affect_dict = {}
        affect_frequencies = Counter()

        emotions = self.distance_calculator.nrc_emotions(
            self.words, threshold=self.threshold
        )

        for word in self.words:
            word_emotions = emotions.get(word, [])
            if len(word_emotions) > 1 or (len(word_emotions) == 1 and word_emotions[0] != "neutral"):
                affect_list.extend(word_emotions)
                affect_dict[word] = word_emotions

        for emotion in affect_list:
            affect_frequencies[emotion] += 1

        sum_values = sum(affect_frequencies.values())
        affect_percent = {
            'fear': 0.0, 'anger': 0.0, 'anticipation': 0.0, 'trust': 0.0,
            'surprise': 0.0, 'positive': 0.0, 'negative': 0.0,
            'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0
        }

        if sum_values > 0:
            for key in affect_frequencies:
                affect_percent[key] = affect_frequencies[key] / sum_values

        self.affect_list = affect_list
        self.affect_dict = affect_dict
        self.raw_emotion_scores = dict(affect_frequencies)
        self.affect_frequencies = affect_percent

    def _top_emotions(self):
        """
        Determine the most prominent emotions in the input.

        Returns
        -------
        list of tuple
            List of (emotion, score) pairs for the highest scoring emotions.
        """
        max_score = max(self.affect_frequencies.values(), default=0)
        return [
            (emotion, score)
            for emotion, score in self.affect_frequencies.items()
            if score == max_score
        ]