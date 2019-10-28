from __future__ import annotations

from collections import Counter
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from gensim.models import KeyedVectors

DIR = os.path.dirname(os.path.realpath(__file__))


class BayesFeatures(BaseEstimator, TransformerMixin):
    """Weights (tf*idf) features with conditional class probabilites.

    Notes
    -----
    Adapted from https://www.kaggle.com/jhoward/.
    """
    def __init__(self) -> None:
        """Set class probabilities."""
        self.r = None

    def pr(self, X, y_i, y) -> list:
        """Calculate single class probabilities."""
        p = X[[int(yi == y_i) for yi in y]].sum(0)
        return (p + 1) / (sum([int(yi) == y_i for yi in y]) + 1)

    def fit(self, X, y) -> BayesFeatures:
        """Calculate and weight all class probabilities."""
        self.r = np.log(self.pr(X, 1, y) / self.pr(X, 0, y))
        return self

    def transform(self, X) -> np.array:
        """Weight with features with class probabilities."""
        return X.multiply(self.r)

    def fit_transform(self, X, y) -> list:
        """Calculate class probabilites and apply as feature weights."""
        self.fit(X, y)
        return self.transform(X)


class WordEmbeddings(BaseEstimator, TransformerMixin):
    """Pre-trained word embedding features.

    Parameters
    ----------
    pre_trained: ``str``, optional (default='')
        Name of pre-trained KeyedVector embedding file in /data.
    emb_mean: ``bool``, optional (default=False)
        Take the mean over all word embeddings to get document vector.
    emb_sum: ``bool``, optional (default=True)
        Take the sum over all word embeddings to get document vector.
    load_only: ``bool``, optional (default=False)
        Don't touch vocab, get full weights from model (for NN input).
    """
    def __init__(self, pre_trained: str = '', emb_mean: bool = False,
                 emb_sum: bool = True, load_only: bool = False) -> None:
        """Initialize vocab (indices), set params."""
        self.vocab = {'<pad>': 0}
        self.i = 0
        self.load_only = load_only
        self.pre_trained = pre_trained
        self.emb_mean = emb_mean
        self.emb_sum = emb_sum
        self.path = f'{DIR}/data/{pre_trained}'
        self.max_len = 0
        self.static_max = False

    def fit(self, X: list = None) -> WordEmbeddings:
        """Load KeyedVectors from path."""
        self.kv = KeyedVectors.load(self.path, mmap='r')
        return self

    def get_matrix(self) -> np.array:
        """Load full embedding matrix from vocab."""
        embedding_matrix = np.zeros((len(self.vocab) + 1, self.kv.vector_size))
        for word, i in self.vocab.items():
            embedding_vector = self.vocab.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def transform(self, X):
        """Transform document into embedded document vectors."""
        from keras.preprocessing.sequence import pad_sequences
        documents = []
        for sentences in X:  # NOTE: vv this could be implemented much neater
            sentence_vector = []
            for word in sentences.split(' '):
                if self.pre_trained and not self.load_only:
                    if word in self.kv:
                        word_vector = self.kv[word]
                    else:  # NOTE: fill with zeros, probably not neat
                        word_vector = np.zeros(self.kv.vector_size)
                else:
                    if not self.vocab.get(word):
                        self.vocab[word] = self.i
                        self.i += 1
                    word_vector = self.vocab[word]
                sentence_vector.append(word_vector)
                if len(sentence_vector) > self.max_len and not self.static_max:
                    self.max_len = len(sentence_vector)

            if not self.load_only:
                if self.emb_mean:
                    sentence_vector = np.mean(sentence_vector, axis=0)
                elif self.emb_sum:
                    sentence_vector = np.sum(sentence_vector, axis=0)
            if self.static_max:
                sentence_vector[:self.max_len]
            documents.append(sentence_vector)

        self.static_max = True
        if not self.pre_trained or self.load_only:
            documents = pad_sequences(documents, maxlen=self.max_len,
                                      dtype='int32',
                                      padding='pre', truncating='pre',
                                      value=0.0)
        return documents

    def fit_transform(self, X: list, y: list = None) -> list:
        """Get embeddings from vocab and compute them given documents."""
        self.fit(X)
        return self.transform(X)


class BertFeatures(BaseEstimator, TransformerMixin):
    """DistilBERT sentence (= document) vectors."""

    def __init__(self) -> None:
        from transformers import DistilBertTokenizer as BertTokenizer
        from transformers import DistilBertModel as BertModel

        pretrained_weights = 'distilbert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.model = BertModel.from_pretrained(pretrained_weights)

    def fit(self, X: list = None) -> BertFeatures:
        """Do nothing."""
        return self

    def transform(self, X: list) -> list:
        """For every document, get (Distil)BERT vector up to 510 tokens."""
        import torch
        documents = []
        for doc in X:
            input_ids = torch.tensor([
                self.tokenizer.encode(doc[:510], add_special_tokens=True)])
            with torch.no_grad():
                output = self.model(input_ids)[0]  # Models out
                documents.append(output[0, 0, :].tolist())
        return documents

    def fit_transform(self, X: list, y: list = None) -> list:
        """Get BERT vectors for all documents."""
        self.fit(X)
        return self.transform(X)


class MajorityBaseline(BaseEstimator, ClassifierMixin):
    """Standard majority baseline implementation using sklearn API."""

    def __init__(self) -> None:
        """Set label counter."""
        self.y_counter = Counter()

    def __str__(self) -> None:
        return "MajorityBaseline"

    def fit(self, X: list, y: list) -> MajorityBaseline:
        """Count the labels."""
        for yi in y:
            self.y_counter[yi] += 1

        return self

    def predict(self, X: list) -> list:
        """Predict the majority label for the provided data."""
        return [self.y_counter.most_common(1)[0][0] for _ in range(len(X))]
