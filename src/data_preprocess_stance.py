import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nrclex import NRCLex
from sklearn.model_selection import train_test_split

def extract_nrc_features(text):
    emotions = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
    scores = NRCLex(text).raw_emotion_scores
    return [scores.get(e, 0) for e in emotions]




def preprocess_data(df, use_nrc=True, include_topic=True, test_size=0.2,
                     random_state=42, sample_level=False):

    required_columns = ['EDU', 'stance']
    if include_topic:
        required_columns.append('topic_id')
        df['topic_id'] = df['topic_id'].fillna('Unknown')
        #
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")

    for idx, (edus, stances) in df[['EDU', 'stance']].iterrows():
        if not isinstance(edus, list) or not isinstance(stances, list):
            raise ValueError(f"Row {idx}: 'EDU' and 'stance' must be lists.")
        if len(edus) != len(stances):
            raise ValueError(f"Row {idx}: 'EDU' and 'stance' lists must be of equal length.")

    all_stances = [label for sublist in df['stance'] for label in sublist]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_stances)

    if sample_level:
        df['label'] = df['stance'].apply(lambda st: label_encoder.transform(st).tolist())

        X = pd.DataFrame()
        X['text'] = df['EDU']  

        if include_topic:
            X['topic_id'] = df['topic_id']

        if use_nrc:
            X['nrc_feats'] = df['EDU'].apply(lambda edus: [extract_nrc_features(e) for e in edus])

        y = df['label']
    else:
        df_exploded = df.explode(['EDU', 'stance'], ignore_index=True)
        df_exploded = df_exploded.rename(columns={'EDU': 'text', 'stance': 'stance_label'})
        df_exploded = df_exploded.dropna(subset=['stance_label'])

        df_exploded['label'] = label_encoder.transform(df_exploded['stance_label'])

        if use_nrc:
            df_exploded['nrc_feats'] = df_exploded['text'].apply(extract_nrc_features)

        X = df_exploded[['text']].copy()
        if include_topic:
            X['topic_id'] = df_exploded['topic_id']
        if use_nrc:
            X['nrc_feats'] = df_exploded['nrc_feats']

        y = df_exploded['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val, 


