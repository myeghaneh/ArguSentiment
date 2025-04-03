"""
Core module for emotion distance calculations.

This module defines the EmotionDistanceCalculator class which computes
the similarity between words based on an emotion lexicon.
"""
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from transformers import pipeline

from ExpandNRC.utils import _compute_similarity, create_emotion_vectors


class EmotionDistanceCalculator:
    """
    Computes emotion similarities between words based on a given lexicon.

    Parameters
    ----------
    emotion_lexicon : dict
        Mapping of words to associated emotions.
    device : str, optional
        Device to use for the transformer model (default is "mps").
    min_similarity : float, optional
        Minimum similarity threshold (default is 0).
    batch_size : int, optional
        Batch size for processing words (default is 32).
    calibration_method : str, optional
        Method to calibrate similarity scores (default is 'exponential').
    temperature : float, optional
        Temperature used in calibration (default is 10.0).
    use_clusters : bool, optional
        Whether to use clustering for similarity (default is False).
    """

    def __init__(self, emotion_lexicon, device="mps", min_similarity=0,
                 batch_size=32, calibration_method='exponential',
                 temperature=10.0, use_clusters=False, cache_dir=None):
        self.normilize_clusters = True
        self.use_clusters = use_clusters
        self.emotion_lexicon = emotion_lexicon
        self.device = device
        self.min_similarity = min_similarity
        self.batch_size = batch_size
        self.calibration_method = calibration_method
        self.temperature = temperature
        self.emotions = ['anger', 'anticipation', 'disgust', 'fear',
                         'joy', 'sadness', 'surprise', 'trust', 'neutral']

        # Set up a user-specific cache directory (default: ~/.cache/lexicon_expansion)
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache",
                                     "ExpandNRC")
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "lexicon_embeddings_cache.pkl")

        self._normalize_lexicon()
        self.embedder = pipeline("feature-extraction",
                                 model="distilbert-base-uncased",
                                 device=self.device)
        self.word_embeddings = {}
        self._cache_lexicon_embeddings()
        self.emotion_vectors = {}

        if self.use_clusters:
            self._initialize_clustering()
            self._precompute_cluster_statistics()

    def _cache_lexicon_embeddings(self):
        """
        Cache embeddings for words in the lexicon. If a valid cache exists in the
        user-specific
        directory, load it. Otherwise, compute embeddings and save them.
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                # Check if the cached lexicon matches the current one
                if cached_data.get("emotion_lexicon") == self.emotion_lexicon:
                    self.word_embeddings = cached_data.get("word_embeddings", {})
                    print(
                        f"Loaded cached embeddings for {len(self.word_embeddings)} "
                        f"words from {self.cache_file}")
                    return
                else:
                    print("Lexicon changed; recalculating embeddings.")
            except Exception as e:
                print("Error loading cache:", e)

        print("Calculating lexicon embeddings...")
        self.lexicon_words = list(self.emotion_lexicon.keys())
        for i in tqdm(range(0, len(self.lexicon_words), self.batch_size)):
            batch = self.lexicon_words[i:i + self.batch_size]
            for word in batch:
                self.word_embeddings[word] = self._get_embedding(word)
        print(f"Cached embeddings for {len(self.word_embeddings)} words")

        # Save the computed embeddings to the user-specific cache directory
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump({
                    "emotion_lexicon": self.emotion_lexicon,
                    "word_embeddings": self.word_embeddings
                }, f)
            print("Cache saved to", self.cache_file)
        except Exception as e:
            print("Error saving cache:", e)

    def _normalize_lexicon(self):
        """Ensure all emotions in the lexicon are stored as lists."""
        normalized_lexicon = {}
        for word, emotions in self.emotion_lexicon.items():
            if isinstance(emotions, str):
                normalized_lexicon[word] = [emotions]
            elif isinstance(emotions, list):
                normalized_lexicon[word] = emotions
            else:
                normalized_lexicon[word] = [str(emotions)]
        self.emotion_lexicon = normalized_lexicon

    def _get_embedding(self, text):
        """Safely obtain an embedding for a single text string."""
        try:
            emb = self.embedder(text)
            return np.mean(emb[0], axis=0).flatten()
        except Exception as e:
            print(f"Error getting embedding for '{text}': {e}")
            return np.zeros(768)

    def _initialize_clustering(self):
        """Initialize PCA and GMM clustering on the cached embeddings."""
        if not self.word_embeddings:
            return
        embeddings_matrix = np.vstack(list(self.word_embeddings.values()))
        self.pca = PCA(n_components=3)
        self.pca_transformed = self.pca.fit_transform(embeddings_matrix)
        self.gmm = GaussianMixture(n_components=3, covariance_type='tied',
                                   random_state=42, n_init=10)
        self.cluster_labels = self.gmm.fit_predict(self.pca_transformed)
        self.cluster_probabilities = self.gmm.predict_proba(self.pca_transformed)
        self.word_clusters = {}
        words = list(self.word_embeddings.keys())
        for word, label, probs in zip(words, self.cluster_labels,
                                      self.cluster_probabilities):
            self.word_clusters[word] = {
                'cluster'      : int(label),
                'probabilities': probs
            }
        self.cluster_words = {i: [] for i in range(3)}
        for word, info in self.word_clusters.items():
            self.cluster_words[info['cluster']].append(word)

    def _precompute_cluster_statistics(self):
        """Precompute similarity statistics for each cluster."""
        self.cluster_stats = {}
        for cluster_id in range(len(self.cluster_words)):
            cluster_words = self.cluster_words[cluster_id]
            cluster_embs = [self.word_embeddings[w] for w in cluster_words]
            n_words = len(cluster_embs)
            sim_matrix = np.zeros((n_words, n_words))
            for i in range(n_words):
                for j in range(i + 1, n_words):
                    sim = _compute_similarity(cluster_embs[i], cluster_embs[j])
                    sim_matrix[i, j] = sim_matrix[j, i] = sim
            self.cluster_stats[cluster_id] = {
                'mean': np.mean(sim_matrix),
                'std' : np.std(sim_matrix)
            }

    def _compute_cluster_aware_similarity(self, word1, emb1, word2, emb2):
        """
        Compute similarity between two words with cluster awareness.
        """
        if not self.use_clusters:
            return _compute_similarity(emb1, emb2)
        cluster1 = self.word_clusters.get(word1, {}).get('cluster', None)
        cluster2 = self.word_clusters.get(word2, {}).get('cluster', None)
        if cluster1 is None or cluster2 is None:
            return _compute_similarity(emb1, emb2)
        if cluster1 != cluster2:
            return -1.0
        return _compute_similarity(emb1, emb2)

    def _compute_fast_normalized_similarity(self, word1, emb1, word2, emb2):
        """
        Compute normalized similarity using precomputed cluster statistics.
        """
        raw_similarity = _compute_similarity(emb1, emb2)
        if word1 not in self.word_clusters:
            transformed_emb = self.pca.transform(emb1.reshape(1, -1))
            cluster_probs = self.gmm.predict_proba(transformed_emb)[0]
        else:
            cluster_probs = self.word_clusters[word1]['probabilities']
        normalized_sims = []
        for cluster_id in range(len(self.cluster_words)):
            stats = self.cluster_stats[cluster_id]
            if stats['std'] > 0:
                norm_sim = (raw_similarity - stats['mean']) / stats['std']
                norm_sim = (norm_sim + 3) / 6
                normalized_sims.append(np.clip(norm_sim, 0, 1))
            else:
                normalized_sims.append(raw_similarity)
        final_sim = np.sum([s * p for s, p in zip(normalized_sims, cluster_probs)])
        return np.clip(final_sim, 0, 1)

    def _calibrate_similarity(self, similarity, method='exponential'):
        """
        Calibrate the similarity score based on the chosen method.
        """
        if method == 'exponential':
            return np.exp(self.temperature * (similarity - 1))
        elif method == 'sigmoid':
            x = self.temperature * (similarity - 0.95)
            return 1 / (1 + np.exp(-x))
        elif method == 'power':
            return similarity ** self.temperature
        elif method == 'minmax':
            if not hasattr(self, '_sim_min'):
                self._sim_min = 0.85
            if not hasattr(self, '_sim_range'):
                self._sim_range = 0.15
            return (similarity - self._sim_min) / self._sim_range
        return similarity

    def process_single_word(self, input_word):
        """
        Process a single word to compute emotion similarities.

        Returns
        -------
        dict
            A dictionary containing distances, similarities, and related data.
        """
        target_emb = self._get_embedding(input_word)
        results = {
            'distances'           : {emotion: float('inf') for emotion in
                                     self.emotions},
            'similarities'        : {emotion: -float('inf') for emotion in
                                     self.emotions},
            'nearest_words'       : {emotion: None for emotion in self.emotions},
            'raw_similarities'    : {emotion: -float('inf') for emotion in
                                     self.emotions},
            'emotion_similarities': {emotion: -float('inf') for emotion in
                                     self.emotions}
        }
        if self.use_clusters and input_word in self.word_clusters:
            results['cluster_info'] = self.word_clusters[input_word]
        emotion_vectors = create_emotion_vectors(self)
        for lex_word, word_emotions in self.emotion_lexicon.items():
            if lex_word not in self.word_embeddings:
                continue
            lex_emb = self.word_embeddings[lex_word]
            if self.use_clusters and self.normilize_clusters:
                raw_similarity = self._compute_fast_normalized_similarity(input_word,
                                                                          target_emb,
                                                                          lex_word,
                                                                          lex_emb)
            else:
                raw_similarity = _compute_similarity(target_emb, lex_emb)
            if raw_similarity >= 0:
                similarity = self._calibrate_similarity(raw_similarity,
                                                        self.calibration_method)
                if similarity >= self.min_similarity:
                    for emotion in word_emotions:
                        if emotion in self.emotions and similarity > \
                                results['similarities'][emotion]:
                            results['distances'][emotion] = 1 - similarity
                            results['similarities'][emotion] = similarity
                            results['raw_similarities'][emotion] = raw_similarity
                            results['nearest_words'][emotion] = lex_word
        if input_word in emotion_vectors:
            target_vector = emotion_vectors[input_word]
            for emotion, nearest_word in results['nearest_words'].items():
                if nearest_word is not None and nearest_word in emotion_vectors:
                    nearest_vector = emotion_vectors[nearest_word]
                    if np.any(target_vector) and np.any(nearest_vector):
                        results['emotion_similarities'][emotion] = _compute_similarity(
                            target_vector, nearest_vector)
                    else:
                        results['emotion_similarities'][emotion] = 0.0
        results['distances'] = {k: np.nan if v == float('inf') else v for k, v in
                                results['distances'].items()}
        results['similarities'] = {k: np.nan if v == -float('inf') else v for k, v in
                                   results['similarities'].items()}
        results['emotion_similarities'] = {k: np.nan if v == -float('inf') else v for
                                           k, v in
                                           results['emotion_similarities'].items()}
        return {input_word: results}

    def process_batch(self, target_words):
        """
        Process a list of words to compute their emotion similarities.

        Returns
        -------
        dict
            Mapping each word to its corresponding result.
        """
        print("Processing words...")
        results = {}
        for word in tqdm(target_words):
            result = self.process_single_word(word)
            results.update(result)
        return results

    def get_cluster_info(self, word):
        """
        Retrieve clustering information for a specific word.

        Parameters
        ----------
        word : str
            The word for which cluster information is needed.

        Returns
        -------
        dict or None
            Cluster information if available; otherwise, None.
        """
        return self.word_clusters.get(word, None)

    def get_cluster_statistics(self):
        """
        Retrieve statistics for each cluster.

        Returns
        -------
        dict
            Cluster statistics including size, sample words, and emotion distribution.
        """
        stats = {}
        for cluster_id in range(3):
            cluster_words = self.cluster_words[cluster_id]
            cluster_emotions = []
            for word in cluster_words:
                if word in self.emotion_lexicon:
                    cluster_emotions.extend(self.emotion_lexicon[word])
            emotion_counts = {}
            for emotion in cluster_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            stats[cluster_id] = {
                'size'                : len(cluster_words),
                'sample_words'        : cluster_words[:5],
                'emotion_distribution': emotion_counts
            }
        return stats

    def get_emotions(self, words, threshold=0.5):
        """
        Return the emotions for a given word or list of words that exceed the given
        similarity threshold.

        Parameters
        ----------
        words : str or list of str
            A single word or a list of words to analyze.
        threshold : float, optional
            The similarity threshold (default is 0.5).

        Returns
        -------
        dict
            Mapping each word to a dictionary of qualifying emotions and their
            similarity scores.
            If no emotions pass the threshold, the word is assigned {'neutral': 0.5}.
        """
        if isinstance(words, str):
            words = [words]
        results = {}
        for word in words:
            word_result = self.process_single_word(word)
            similarities = word_result.get(word, {}).get('similarities', {})
            valid_emotions = {}
            for emotion, sim in similarities.items():
                try:
                    if not np.isnan(sim) and sim >= threshold:
                        valid_emotions[emotion] = sim
                except Exception:
                    pass
            if not valid_emotions:
                valid_emotions['neutral'] = 0.5
            results[word] = valid_emotions
        return results

    def nrc_emotions(self, words, threshold=0.5):
        """
        Transform the output of get_emotions to return only the emotion labels for
        each word.

        Parameters
        ----------
        words : str or list of str
            A single word or a list of words to analyze.
        threshold : float, optional
            The similarity threshold (default is 0.5).

        Returns
        -------
        dict
            A dictionary mapping each word to a list of emotion labels that exceed
            the threshold.
            If no emotion passes the threshold for a word, it returns ['neutral'].
        """
        # Get the emotion similarity scores using the get_emotions method.
        emotions_with_values = self.get_emotions(words, threshold=threshold)

        # Transform the output to return only the keys (emotion labels)
        nrc_emotions_dict = {}
        for word, emotions in emotions_with_values.items():
            # Ensure that if no valid emotion is found, we get a list with 'neutral'
            if not emotions:
                nrc_emotions_dict[word] = ['neutral']
            else:
                nrc_emotions_dict[word] = list(emotions.keys())
        return nrc_emotions_dict
