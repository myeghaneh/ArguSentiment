from typing import List, Tuple

import pandas as pd
import torch
from nrclex import NRCLex
from tqdm import tqdm

from ExpandNRC.emotion_frequences import EmotionFrequencyCalculator
from data_preprocess_stance import preprocess_data
from train_stance import train_model


class EmotionVectorFactory:
    _ORDER = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
              'sadness', 'surprise', 'trust', 'positive', 'negative']

    def __init__(self, lexicon_path: str, threshold: float = 0.6, device: str = "cpu"):
        lexicon = NRCLex(lexicon_path).lexicon  #__lexicon__
        self._calc = EmotionFrequencyCalculator(lexicon, threshold=threshold,
                                                device=device)

    def vector(self, text: str) -> List[float]:
        self._calc.load_raw_text(text)
        freqs = self._calc.affect_frequencies
        return [freqs.get(e, 0.0) for e in self._ORDER]


def preprocess_with_expandnrc(df: pd.DataFrame, lexicon_path: str,
                              include_topic: bool = True, **kwargs) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    device = kwargs.pop('device', 'mps')
    threshold = kwargs.pop('threshold', 0.6)
    test_size = kwargs.pop('test_size', None)
    random_state = kwargs.pop('random_state', None)
    sample_level = kwargs.pop('sample_level', None)
    X_tr, X_val, y_tr, y_val = preprocess_data(
        df,
        use_nrc=True,
        include_topic=include_topic,
        test_size=test_size,
        random_state=random_state,
        sample_level=sample_level
    )
    evf = EmotionVectorFactory(lexicon_path,
                               device=device,
                               threshold=threshold)
    for split, name in zip((X_tr, X_val), ('train', 'val')):
        print(split['text'].head())
        print(type(split['text'].iloc[0]))
        print(split['text'].iloc[0])

        desc = f"Computing ExpandNRC features ({name})"
        # Use a list comprehension with tqdm to show progress
        # split['expnrc_feats'] = [evf.vector(text) for text in tqdm(split['text'], desc=desc)] 
        split['expnrc_feats'] = [evf.vector(" ".join(text) if isinstance(text, list) else text) 
                         for text in tqdm(split['text'], desc=desc)]  #Test
    
    return X_tr, X_val, y_tr, y_val


