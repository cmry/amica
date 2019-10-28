"""Main experiment file."""

import warnings; warnings.filterwarnings("ignore")

# reproducibility bit ----------------
from random import seed; seed(42)
from numpy.random import seed as np_seed; np_seed(42)
from tensorflow.compat.v1 import set_random_seed; set_random_seed(42)
import os; os.environ['PYTHONHASHSEED'] = str(42)
# -----------------------------------

import argparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from evaluation import Evaluation
from models import (BayesFeatures, BertFeatures, WordEmbeddings)
from reader import Reader, merge_datasets
from utils import debug_tests


class EnglishCompare(object):

    def __init__(self, pipeline: Pipeline, datasets: list = None,
                 merge: bool = False, cross: bool = True, neural: bool = False,
                 clean: bool = True, preprocess: bool = False,
                 multi_read: int = 0) -> None:
        # NOTE: comment out those unavailable, or provide own list
        self.datasets = [
            ('bretschneider', 'agg_set'),
            ('kaggle', 'kag_set'),
            ('kontostathis', 'msp_set'),
            ('maral', 'ytb_set'),
            ('vanhee', 'asken_set'),
            ('xu', 'xu_set'),
            ('kaggle', 'kag_conv'),
            ('vanhee', 'asken_conv'),
            ('toxic', 'toxic_set')
        ] if not datasets else datasets
        self.data = Reader(clean=True, preprocess=False, language='en',
                           multi_threading=multi_read)
        self.eval = Evaluation(pipeline,
                               headers=['_'.join(x) for x in self.datasets],
                               merge=merge, cross=cross, neural=neural)
        self.merge = merge

    def _cross_data(self) -> (list, list):
        train = [data for data in self.data.subset(self.datasets)]
        test = [_data for _data in self.data.subset(self.datasets)]
        if self.merge:
            train = train[:-1]
            train = [merge_datasets(train)]
        return train, test

    def run(self, nest: bool = False, store: bool = False,
            report: bool = False) -> None:
        print(f"\n> Merging: {self.merge}")
        train, test = self._cross_data()
        self.eval.score(train, test=test, nest=nest, store=store,
                        report=report)


class DutchCompare(object):

    def __init__(self, pipeline: Pipeline, datasets: list = None,
                 merge: bool = False, cross: bool = True, neural: bool = False,
                 clean: bool = True, preprocess: bool = False,
                 multi_read: int = 0) -> None:
        if merge:
            raise(ValueError("Sorry, the NL data manually does merging."))
        self.datasets = [
            ('vanhee', 'asknl_set'),
            ('vanhee', 'simnl_set'),
            ('vanhee', 'donnl_set'),
            ('vanhee', 'cnvnl_set'),
            ('vanhee', 'cnsnl_set')
        ] if not datasets else datasets
        self.data = Reader(clean=True, preprocess=False, language='nl')
        headers = ['_'.join(x) for x in self.datasets]
        headers += [f'{headers[0]}+{headers[1]}']
        self.eval = Evaluation(pipeline, headers=headers, cross=cross,
                               neural=neural)

    def _combine_sets(self) -> dict:
        sets = {'ask': '', 'sim': '', 'don': '', 'cnv': '', 'cns': ''}
        train = [data for data in self.data.subset(self.datasets)]

        for data in train:
            for key in sets:
                if key in data.id:
                    sets[key] = data
        return sets

    def run(self, nest: bool = False, store: bool = False,
            report: bool = False) -> None:
        dsets = self._combine_sets()
        config = [
            (dsets['ask'], dsets['ask']),
            (dsets['sim'], dsets['sim']),
            (dsets['cnv'], dsets['cnv']),
            (dsets['cns'], dsets['cns']),
            (dsets['ask'], dsets['sim']),
            (dsets['sim'], dsets['ask']),
            (dsets['ask'], dsets['don']),
            (dsets['sim'], dsets['don']),
            (dsets['ask'], dsets['cnv']),
            (dsets['ask'], dsets['cns']),
            (dsets['sim'], dsets['cnv']),
            (dsets['sim'], dsets['cns']),
            (dsets['cnv'], dsets['ask']),
            (dsets['cnv'], dsets['sim']),
            (dsets['cnv'], dsets['don']),
            (dsets['cnv'], dsets['cns']),
            (dsets['cns'], dsets['cnv']),
            (dsets['cns'], dsets['ask']),
            (dsets['cns'], dsets['sim']),
            (dsets['cns'], dsets['don']),
            (merge_datasets([dsets['ask'], dsets['sim']]), dsets['ask']),
            (merge_datasets([dsets['ask'], dsets['sim']]), dsets['sim']),
            (merge_datasets([dsets['ask'], dsets['sim']]), dsets['don']),
            (merge_datasets([dsets['ask'], dsets['sim']]), dsets['cnv']),
            (merge_datasets([dsets['ask'], dsets['sim']]), dsets['cns']),
            (merge_datasets([dsets['ask'], dsets['sim'], dsets['cnv'],
                             dsets['cns']]), dsets['ask']),
            (merge_datasets([dsets['ask'], dsets['sim'], dsets['cnv'],
                             dsets['cns']]), dsets['sim']),
            (merge_datasets([dsets['ask'], dsets['sim'], dsets['cnv'],
                             dsets['cns']]), dsets['don']),
            (merge_datasets([dsets['ask'], dsets['sim'], dsets['cnv'],
                             dsets['cns']]), dsets['cnv']),
            (merge_datasets([dsets['ask'], dsets['sim'], dsets['cnv'],
                             dsets['cns']]), dsets['cns'])
        ]
        for train, test in config:
            self.eval.score([train], test=[test], nest=nest, df=False)


def select_model(key: str) -> Pipeline:
    """Select the model to use based on argparse input."""

    # NOTE: all these if statements don't look particularly nice, but we also
    # don't want to load a bunch of models we're not gonna use now, do we?

    # Simple Default Test
    if key == 'debug':
        return {
            ('vect', TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                     max_df=0.9, use_idf=1, smooth_idf=1,
                                     sublinear_tf=1)): {},
            ('nbf', BayesFeatures()): {},
            ('lr', LogisticRegression(dual=True, random_state=42,
                                      class_weight="balanced")): {}
        }

    # FINAL BINARY SVM BASELINE
    elif key == 'baseline':
        return {
            ('vect', CountVectorizer(binary=True)): {
                'vect__ngram_range': [(1, 1), (1, 2), (1, 3)]
            },
            ('svc', LinearSVC(random_state=42)): {
                'svc__C': [1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3],
                'svc__loss': ['hinge', 'squared_hinge'],
                'svc__class_weight': [None, "balanced"]
            }
        }

    elif key == 'debug-baseline':
        return {
            ('vect', CountVectorizer(binary=True)): {},
            ('svc', LinearSVC(random_state=42)): {}
        }

    # NB-SVM Model
    elif key == 'nbsvm':
        return {
            ('vect', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                                     use_idf=1, smooth_idf=1,
                                     sublinear_tf=1)): {},
            ('nbf', BayesFeatures()): {},
            ('lr', LogisticRegression(dual=True, solver='liblinear',
                                      random_state=42,
                                      class_weight="balanced")): {
                'lr__C': [1, 2, 3, 4, 5, 10, 25, 50, 100, 200, 500]
            }
        }

    elif key == 'debug-nbsvm':
        return {
            ('vect', TfidfVectorizer()): {},
            ('nbf', BayesFeatures()): {},
            ('lr', LogisticRegression()): {}
        }

    # LR over Embeddings
    elif key == 'w2v':
        return {
            # NOTE: cow.nl.kv for Dutch
            ('vct', WordEmbeddings(pre_trained="w2v.kv")): {},
            ('lr', LogisticRegression(class_weight="balanced",
                                      solver='liblinear', random_state=42)): {
                'lr__C': [1, 2, 3, 4, 5, 10, 25, 50, 100, 200, 500],
            }
        }

    elif key == 'debug-w2v':
        return {
            # NOTE: cow.nl.kv for Dutch
            ('vct', WordEmbeddings(pre_trained="w2v.kv")): {},
            ('lr', LogisticRegression()): {}
        }

    # LR over DistilBert
    elif key == 'bert':
        return {
            ('vect', BertFeatures()): {},
            ('lr', LogisticRegression(class_weight="balanced",
                                      solver='liblinear', random_state=42)): {
                'lr__C': [1, 2, 3, 4, 5, 10, 25, 50, 100, 200, 500],
            }
        }

    elif key == 'debug-bert':
        return {
            ('vect', BertFeatures()): {},
            ('lr', LogisticRegression()): {}
        }

    # Reproduction B-LSTM
    elif key == 'blstm':
        from neural import ReproductionNeuralNetwork
        return {
            ('neur', ReproductionNeuralNetwork(
                m_type='blstm', inp_dim=128, num_classes=2, learn_rate=0.01,
                batch_size=128, epochs=10, embed_size=50)): {}
        }

    elif key == 'debug-blstm':
        from neural import ReproductionNeuralNetwork
        return {
            ('neur', ReproductionNeuralNetwork(
                m_type='blstm', inp_dim=128, num_classes=2, learn_rate=1,
                batch_size=32, epochs=1, embed_size=50)): {}
        }

    # Reproduction CNN
    elif key == 'cnn':
        from neural import ReproductionNeuralNetwork
        return {
            ('neur', ReproductionNeuralNetwork(
                m_type='cnn', inp_dim=128, num_classes=2, learn_rate=0.01,
                batch_size=128, epochs=10, embed_size=50)): {}
        }

    elif key == 'debug-cnn':
        from neural import ReproductionNeuralNetwork
        return {
            ('neur', ReproductionNeuralNetwork(
                m_type='cnn', inp_dim=128, num_classes=2, learn_rate=1,
                batch_size=32, epochs=1, embed_size=50)): {}
        }

    # Own NN Grid
    elif key == 'nn':
        from neural import ReproductionNeuralNetwork
        return {
            ('neur', ReproductionNeuralNetwork(
                m_type='clstm', inp_dim=128, num_classes=2, learn_rate=0.01,
                batch_size=128, epochs=10, embed_size=50, character_level=True,
                early_stop=3)
             ): {
                # NOTE: roughly optimal params: 128 batch / 100, 50 embeddings
                'neur__batch_size': [32, 64, 128, 256],
                'neur__embed_size': [50, 100, 200, 300],
                'neur__learn_rate': [0.1, 0.01, 0.05, 0.001, 0.005]
            }
        }

    elif key == 'debug-nn':
        from neural import ReproductionNeuralNetwork
        return {
            ('neur', ReproductionNeuralNetwork(
                m_type='clstm', inp_dim=128, num_classes=2, learn_rate=1,
                batch_size=32, epochs=1, embed_size=50)): {}
        }

    else:
        raise(KeyError(f"Sorry, `{key}` is not a valid --model name."))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
             description='Cyberbullying detection replication environment.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model',
                        help="""debug | baseline | nbsvm | w2v | bert |
                         blstm | cnn | nn -- Debug will run all possible
                         configurations!""", type=str)
    parser.add_argument('--language', default='en', type=str, help="""Run on
                        English (en) or Dutch (nl) data.""")
    parser.add_argument('--preprocessing', default='clean', type=str,
                        help="none | clean | preprocess")
    parser.add_argument('--merge', default=False, type=bool, help="""Merge all
                        training sets (D_All in paper).""")
    parser.add_argument('--nest', default=False, type=bool, help="""Report
                        nested cross-validation scores (only relevant when
                        using GridSearch).""")
    parser.add_argument('--single_domain', default=False, type=bool,
                        help="Don't run eval cross-domain.")
    parser.add_argument('--multi_read', default=0, help="""Number of cores the
                        _reader_ should use for multi-threading.""")
    parser.add_argument('--store', default=False, type=bool, help="""Save the
                        best model in a pickle file under /results.""")
    parser.add_argument('--report', default=False, type=bool, help="""Report
                        the most important features for SVM/LR models.""")
    args = parser.parse_args()

    Experiment = EnglishCompare if args.language == 'en' else DutchCompare
    if args.model == 'debug':
        debug_tests(args, EnglishCompare, select_model)
    else:
        Experiment(pipeline=select_model(args.model), merge=args.merge,
                   datasets=None, cross=args.single_domain,
                   neural=args.model in ['blstm', 'cnn', 'nn'],
                   clean='clean' in args.preprocessing,
                   preprocess='preprocess' in args.preprocessing,
                   multi_read=args.multi_read).run(nest=args.nest,
                                                   store=args.store,
                                                   report=args.report)
