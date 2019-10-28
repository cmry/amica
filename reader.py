"""Corpora reader and split tools."""

from __future__ import annotations

import csv
import re
import sys
from copy import deepcopy
from glob import glob
from multiprocessing import Pool
from os import path, environ as env

import emot
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# NOTE: might want to comment this out / change it, it's for spaCy
env["MKL_NUM_THREADS"] = "20"
env["MKL_DOMAIN_NUM_THREADS"] = "MKL_BLAS=20"
env["OMP_NUM_THREADS"] = "1"
env["MKL_DYNAMIC"] = "FALSE"
env["OMP_DYNAMIC"] = "FALSE"

DIR = path.dirname(path.realpath(__file__))


class Dataset(object):
    """Dataset wrapper class providing simplified interfacing.

    Parameters
    ----------
    author: ``str``, required
        Author string ID (corresponds to directory in /corpora).
    subset: ``str``, required
        Set string ID (corresponds to .csv file in /corpora).
    """
    def __init__(self, author: str, subset: str) -> None:
        self.author = author
        self.subset = subset
        self.id = f'{author}_{subset}'
        self.data = None
        self.labels = None
        self.train = [[], []]
        self.test = [[], []]

    def _unique(self, X: list, y: list) -> (str, str):
        """Filter any possible non-unique instances (safety 1st)."""
        X_set = set()
        for Xi, yi in zip(X, y):
            if Xi not in X_set:
                X_set.add(Xi)
                yield Xi, yi

    def add(self, X: list, y: list) -> Dataset:
        """Split provided data. Set class splits and labels to attributes."""
        X, y = zip(*self._unique(X, y))
        self.train[0], self.test[0], self.train[1], self.test[1] = \
            train_test_split(X, y, random_state=42, stratify=y, test_size=0.1)

        self.data = self.train[0] + self.test[0]
        self.labels = self.train[1] + self.test[1]
        self.train = tuple(self.train)
        self.test = tuple(self.test)
        return self

    def get(self, split):
        """Dictionary-style get data split (train or test)."""
        return eval(f'self.{split}')


def merge_datasets(data_list: list) -> Dataset:
    """Given a list of Dataset objects, return one merged version."""
    main_data = deepcopy(data_list.pop(0))
    main_data.author = [main_data.author]
    main_data.subset = [main_data.subset]
    main_data.id = [main_data.id]
    main_data.train = list(main_data.train)
    main_data.test = list(main_data.test)

    for data in data_list:
        main_data.author.append(data.author)
        main_data.subset.append(data.subset)
        main_data.id.append(data.id)
        main_data.data += data.data
        main_data.labels += data.labels
        main_data.train[0] += data.train[0]
        main_data.test[0] += data.test[0]
        main_data.train[1] += data.train[1]
        main_data.test[1] += data.test[1]

    main_data.train = tuple(main_data.train)
    main_data.test = tuple(main_data.test)

    return main_data


class Reader(object):
    """Reader of Datasets and processor of those data.

    Parameters
    ----------
    ddir: ``str``, optional (default=directory_to_repo/corpora)
        The data directory. Change if using own data directory.
    clean: ``bool``, optional (default=True)
        If simple tokenization and special character removing should be done.
        Default for the paper is True (as demonsrated).
    preprocess: ``bool``, optional (default=False)
        Intelligent handling of social tokens and lemmatization (+ clean).
    language: ``str``, optional (default=en)
        Switch parser from English (en) to Dutch (nl).
    multi_threading: ``int``, optional (default=0)
        Read datasets over multiple threads. Specify number of threads to use.
    """
    def __init__(self, ddir: str = f'{DIR}/corpora', clean: bool = True,
                 preprocess: bool = False, language: str = 'en',
                 multi_threading: int = 0) -> None:
        """Load spacy, get data from dirs."""
        self.dirs = {}  # NOTE: v always clean if preprocess
        self.clean, self.proc = preprocess if preprocess else clean, preprocess

        lang = 'en_core_web_sm' if language == 'en' else 'nl_core_news_sm'
        self.nlp = spacy.load(lang, disable=['parser', 'tagger', 'ner'])

        for l in glob(ddir + '/**/**/*.csv') + glob(ddir + '/**/*.csv'):
            ls = l.split('/')
            if not self.dirs.get(ls[-3]):
                self.dirs[ls[-3]] = {}
            self.dirs[ls[-3]][l.split('/')[-1].replace('.csv', '')] = l
        self.multi_thread = multi_threading

    def __str__(self) -> str:
        """Dump the available datasets."""
        out = ''
        for k, v in sorted(self.dirs.items()):
            out += '\n\n' + k
            for ki in sorted(v.keys()):
                out += '\n\t\t' + ki
        return out

    @staticmethod
    def is_emo(word: str) -> bool:
        """Use emot to detect if something is an emoticion or emoji."""
        emoji = emot.emoji(word)['flag']
        emoticon = emot.emoticons(word)
        try:
            emoticon = emoticon['flag']
        except TypeError:
            emoticon = emoticon[0]['flag']
        return bool(emoji) + bool(emoticon)

    def preprocess(self, text: str) -> str:
        """Wrapper function for both clean and preprocessing methods."""
        if not self.clean:
            text = [token.text for token in self.nlp(text)]
            clean_text = ' '.join(text)
        else:
            if self.proc:
                text = [token.lemma_ for token in self.nlp(text)]
                valid_tokens = []
                for i, token in enumerate(text):
                    try:
                        if token[i - 1] == token[i]:
                            continue
                    except IndexError:
                        pass
                    if self.is_emo(token):
                        valid_tokens.append('__EMO__')
                    elif token.startswith('@'):
                        valid_tokens.append('__AT__')
                    elif token.startswith('#'):
                        valid_tokens.append('__HT__')
                    else:
                        valid_tokens.append(token)

                text = ' '.join(valid_tokens)
            else:
                tokens = [token.text for token in self.nlp(text)]
                text = ' '.join(tokens)
            clean_text = re.sub(r'([^\s\w])+', '', text)

        return clean_text.lower()

    def load(self, key: str, value: str) -> (str, str):
        """Load wrapper, also does pooling (aka multi-threading)."""
        reader = csv.reader(open(self.dirs[key][value]))
        l_index, c_index = 0, 1

        X, y = zip(*[(line[c_index], line[l_index]) for line in reader])
        if self.multi_thread:
            with Pool(self.multi_thread) as p:
                X = p.map(self.preprocess, X)
        else:
            X = [self.preprocess(x) for x in X]
        for Xi, yi in zip(X, y):
            yield Xi, yi

    def subset(self, datasets: list) -> Dataset:
        """Subset the provided datasets."""
        for pub, dset in tqdm(datasets, file=sys.stdout):
            yield Dataset(pub, dset).add(*zip(*self.load(pub, dset)))
