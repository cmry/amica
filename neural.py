"""Neural models."""

# from __future__ import annotations
# NOTE: this requires python 3.7. You'll have to comment out
# -> VocabularyProcessor and -> ReproductionNeuralNetwork in their `def fit()`
# if you don't care for the class type annotations (or, more likely, upgrading
# Python)

# this mess is necessary but makes linters cry ------------
from os import environ as env; env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; env['CUDA_VISIBLE_DEVICES'] = '0,1'
import tensorflow as tf; env['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ----------------------------------------------------------

from collections import Counter
from typing import Callable

import numpy as np

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.engine.input_layer import Input
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional
from keras.layers import concatenate, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.engine.topology import Layer
from sklearn.base import BaseEstimator, ClassifierMixin
from tflearn.data_utils import to_categorical, pad_sequences
# FIXME: ^ should be replaced; tflearn is quite the dependency for simple
# padding, but is required to reproduce accurately


class VocabularyProcessor(object):
    """Adaptation of tflearn's VocabularyProcessor.

    Parameters
    ----------
    max_len: ``int``, optional (default=128)
        Maximum sequence length.
    min_freq: ``int``, optional (default=1)
        Minimum token frequency.
    vocab_len: ``int`` optional (default=None)
        Maximum vocabulary length.
    chars: ``bool``optional (default=False)
        Split tokens into characters for character modeling.

    Notes
    -----
    This class was deprecated, but required for reproducibility.
    """
    def __init__(self, max_len: int = 128, min_freq: int = 1,
                 vocab_len: int = None, chars: bool = False) -> None:
        """Set class attributes."""
        self.vocab = Counter()
        self.max_len = max_len
        self.min_freq = min_freq
        self.vocab_len = vocab_len
        self.chars = chars

    def process(self, doc: str) -> str:
        """Process sentence, lowercase, cut to max length, character split."""
        sent = [token.lower() for token in doc.split(' ')][:self.max_len]
        if self.chars:
            sent = [char for char in sent]
        return sent

    def fit(self, documents: list):  # -> VocabularyProcessor:
        """Fit (and optionally restrict) vocabulary."""
        for doc in documents:
            for w in self.process(doc):
                self.vocab[w] += 1
        if not self.vocab_len:
            self.vocab_len = len(self.vocab)
        self.vocab = {w: ix + 1 for ix, (w, freq) in enumerate(dict(
                          self.vocab.most_common(self.vocab_len)
                      ).items()) if freq >= self.min_freq}
        return self

    def transform(self, documents: list) -> list:
        """Convert documents into vocab indices."""
        batch = []
        for doc in documents:
            indices = [self.vocab.get(word, 0) for word in self.process(doc)]
            batch.append([x for x in indices if x])
        return batch

    def fit_transform(self, documents: list) -> list:
        """Fit and convert documents to vocab."""
        self.fit(documents)
        return self.transform(documents)


class AttLayer(Layer):
    """Tensorflow-compatible attention layer (over Theano).

    Notes
    -----
    Source: github.com/richliao/textClassifier/issues/13#issuecomment-380695955

    Please note that we did not use this in the papers' experiments, it's just
    here for completeness.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize layer."""
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape: tuple) -> None:
        """Set the weight layer (randomly)."""
        super(AttLayer, self).build(input_shape)
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[-1],),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Weight input layer x by attention W."""
        eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))

        ai = K.exp(eij)
        weights = ai / K.expand_dims(K.sum(ai, axis=1), 1)

        weighted_input = x * K.expand_dims(weights, 2)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """Convert 3D input into 2D output shape."""
        return (input_shape[0], input_shape[-1])


class ReproductionNeuralNetwork(BaseEstimator, ClassifierMixin):
    """Neural architectures for reproduction + grid search / early stop.

    Parameters
    ----------
    m_type: ``str``, required
        Model type: [cnn] | [clstm] | lstm | blstm | blstmatt -- models in
        brackets are used in the paper.
    inp_dim: ``int``, required
        Input dimensions (usually max length).
    num_classes: ``int```, required
        Number of classes in the output (e.g., 2 for binary).
    learn_rate: ``float``, optional (default=0.01)
        Learning rate for ADAM.
    batch_size: ``int``, optional (default=32)
        Batch size.
    epochs: ``int``, optional (default=4)
        Number of epochs to run the network.
    embed_size: ``int``, optional (default=50)
        Size (dimensionality) of the embedding layer.
    character_level: ``bool``, optional (default=False)
        Run model on character level (rather than token level).
    early_stop: ``int``, optional (default=0)
        If not zero, splits off a development set, and applies early stopping
        with a patience of n epochs of no improvement (where n is the int)
        in development loss. 

    Notes
    -----
    There is no pre-trained option for the embeddings, as this was not required
    for the reproduction (implying it was not necessary).
    """
    def __init__(self,
                 m_type: str,
                 inp_dim: int,
                 num_classes: int,
                 learn_rate: float = 0.01,
                 batch_size: int = 32,
                 epochs: int = 4,
                 embed_size: int = 50,
                 character_level: bool = False,
                 early_stop: int = 0):
        """Set all params, including processor and fitted."""

        self.m_type = m_type
        self.max_len = inp_dim
        self.num_classes = num_classes
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model, self.vocab_size = (None, ) * 2
        self.embed_size = embed_size
        self.processor = VocabularyProcessor(
            max_len=self.max_len, min_freq=2, vocab_len=10000,
            chars=character_level)
        self.fitted = False
        self.early_stop = early_stop

    def convert(self, X, y=None):
        """Pad and index X, make y categorical."""
        X = np.array(list(self.processor.transform(X)))
        X = pad_sequences(X, maxlen=self.max_len, value=0.)
        if y:
            y = np.asarray(y)
            y = to_categorical(y, nb_classes=self.num_classes)
            return X, y
        else:
            return X

    def fit(self, X: list, y: list): # -> ReproductionNeuralNetwork:
        """Fit vocab and model & if necessary, split off dev for early stop."""
        if not self.fitted:  # if the vocab isn't fitted to index tokens
            self.processor.fit(X)
            self.fitted = True
        X, y = self.convert(X, y)

        if self.early_stop:  # dev splitting -- not pretty, but it works
            Xa = X[:int(len(X) * 0.9)]
            ya = y[:int(len(y) * 0.9)]
            Xv = X[-int(len(X) * 0.1):]
            yv = y[-int(len(y) * 0.1):]
            X, y = np.array(Xa), np.array(ya)
            validation_data = np.array(Xv), np.array(yv)
            callbacks = [EarlyStopping(monitor='val_loss',
                                       patience=self.early_stop)]
        else:
            callbacks = None
            validation_data = None

        self.model = self.text_net(
            eval(f"self.{self.m_type}"), self.vocab_size, self.embed_size,
            self.num_classes, self.learn_rate, self.max_len)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                       shuffle=True, verbose=1, callbacks=callbacks,
                       validation_data=validation_data)
        return self

    def predict(self, X: list) -> list:
        """Return argmaxed predictions as strings."""
        X = self.convert(X)
        ŷ = np.argmax(self.model.predict(X), axis=1)
        return [str(int(yi)) for yi in ŷ]

    # NOTE: could've also placed the dropout layers in a wrapper function, but
    # didn't want to stray too much from Agrawal et al. for clarity's sake.
    def text_net(self, net: Callable[[Embedding, int], tf.Tensor],
                 vocab_size: int, embed_size: int, num_classes: int,
                 learn_rate: float, max_len: int):
        """General text classification arch wrapper for different layers."""
        _input = Input(shape=(max_len,), dtype='int32')
        emb = Embedding(self.processor.vocab_len, embed_size,
                        input_length=max_len, trainable=True)(_input)
        outs = net(emb, embed_size)
        preds = Dense(self.num_classes, activation='softmax')(outs)
        model = Model(_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def cnn(self, emb: Embedding, embed_size: int) -> tf.Tensor:
        """Convolutional Neural Network."""
        x = Dropout(0.25)(emb)
        x1 = Conv1D(embed_size, 3, padding='valid', kernel_regularizer=l2(.01),
                    activation='relu')(x)
        x2 = Conv1D(embed_size, 4, padding='valid', kernel_regularizer=l2(.01),
                    activation='relu')(x)
        x3 = Conv1D(embed_size, 5, padding='valid', kernel_regularizer=l2(.01),
                    activation='relu')(x)
        x = concatenate([x1, x2, x3], axis=1)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        return x

    def lstm(self, emb: Embedding, embed_size: int) -> tf.Tensor:
        """Long Short-Term Memory Nework."""
        x = Dropout(0.25)(emb)
        x = LSTM(embed_size)(x)
        x = Dropout(0.50)(x)
        return x

    def blstm(self, emb: Embedding, embed_size: int) -> tf.Tensor:
        """Bi-directional Long Short-Term Memory Network."""
        x = Dropout(0.25)(emb)
        x = Bidirectional(LSTM(embed_size))(x)
        x = Dropout(0.50)(x)
        return x

    def blstmatt(self, emb: Embedding, embed_size: int) -> tf.Tensor:
        """Bi-directional Long Short-Term Memory Network with attention."""
        x = Dropout(0.25)(emb)
        x = Bidirectional(LSTM(embed_size, return_sequences=True))(x)
        x = AttLayer()(x)
        x = Dropout(0.50)(x)
        return x

    def clstm(self, emb: Embedding, embed_size: int) -> tf.Tensor:
        """Convolutional Long Short-Term Memory Network.

        Notes
        -----
        From https://github.com/bicepjai/Deep-Survey-Text-Classification/.
        """
        convs = []
        for filter_size in [10, 20, 30, 40, 50]:
            convs.append(
                Conv1D(filters=embed_size, kernel_size=filter_size,
                       padding='valid', kernel_regularizer=l2(.01),
                       activation='relu')(emb))
        x = concatenate(convs, axis=1)
        x = LSTM(64, return_sequences=False)(x)
        x = Dense(128, activation="relu")(x)
        return x
