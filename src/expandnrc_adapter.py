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
        lexicon = NRCLex(lexicon_path).__lexicon__
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
        use_nrc=False,
        include_topic=include_topic,
        test_size=test_size,
        random_state=random_state,
        sample_level=sample_level
    )
    evf = EmotionVectorFactory(lexicon_path,
                               device=device,
                               threshold=threshold)
    for split, name in zip((X_tr, X_val), ('train', 'val')):
        desc = f"Computing ExpandNRC features ({name})"
        # Use a list comprehension with tqdm to show progress
        split['expnrc_feats'] = [evf.vector(text) for text in tqdm(split['text'], desc=desc)]
    return X_tr, X_val, y_tr, y_val


if __name__ == "__main__":
    df = pd.read_json("../data/dfIBM_stance-v1.json")
    print(df.columns)
    X_tr, X_val, y_tr, y_val = preprocess_with_expandnrc(
        df,
        "../NRCLex/nrc_v3.json",
        test_size=0.2,
        random_state=42,
        sample_level=False,
        device='mps',
        threshold=0.6
    )
    print("continue with training")
    model, history, _ = train_model(
        (X_tr['text'].tolist(), X_tr['topic_id'].tolist(),
         X_tr['expnrc_feats'].tolist()),
        y_tr,
        (X_val['text'].tolist(), X_val['topic_id'].tolist(),
         X_val['expnrc_feats'].tolist()),
        y_val,
        device="mps" if torch.cuda.is_available() else "cpu",
        sample_level=False
    )
