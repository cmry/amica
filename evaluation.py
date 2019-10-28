"""Evaluation of experiments."""

import pickle
from collections import ChainMap
from copy import deepcopy

import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score as cvs
from sklearn.pipeline import Pipeline

from reader import Dataset
from utils import sk_clf_report


F1 = make_scorer(f1_score, average='binary', pos_label='1')


class Evaluation(object):
    """Pipeline evaluation, under given data conditions.

    This class handles all structuring of data, fitting of the provided
    pipeline and logging (and displaying) the output of the evaluations. It's
    tightly integrated with the rest, probably would be better if it had all
    private methods, but hey.

    Parameters
    ----------
    pipeline: ``dict``, required
        Sklearn-style pipeline dictionary. Keys are tuples with (id names
        (str), sklearn-API class (obj)), and values another dictionary, with
        {parameter name (str): [list with parameter values (str)]}.
    headers: ``lists``, required
        List with string identifiers for the datasets to be included in
        the final results table.
    merge: ``bool``, optional (default=False)
        If the datasets should be merged into one big train set, testing will
        still be done on individual test sets.
    cross: ``bool``, optional (default=True)
        If the datasets should be evaluated cross-domain (should stay on for
        paper experiments).
    neural: ``bool``, optional (default=False)
        If this is a neural set-up where train/dev/test is required and no
        10-fold cross validation is conducted.
    """
    def __init__(self, pipeline: dict, headers: list, merge: bool = False,
                 cross: bool = True, neural: bool = False) -> None:
        """Build score record and results table."""
        self.scores = {}
        self.df = pd.DataFrame([], index=headers if cross else None,
                               columns=headers)
        self.merge = merge
        self.cross = cross
        self.neural = neural
        self.pipeline = pipeline

    def _add(self, data_id: str, _data_id: str, scores: list) -> None:
        """Add score for train data_id and test _data_id to results table."""
        try:  # NOTE: because it doesn't work for nl
            self.scores[data_id].append(scores)
            self._df_cross_add(scores, data_id, _data_id)
        except TypeError:
            pass

    def _create(self, data_id: int) -> None:
        """Create data_id entry in results table."""
        if isinstance(data_id, list):
            data_id = '_'.join(data_id)
        if data_id not in self.scores:
            self.scores[data_id] = []

    def _df_cross_add(self, score: int, index: str, column: str) -> None:
        """Add value to pair-wise results table."""
        self.df[column][index] = score

    def _oversample(self, X: list, y: list, factor: int = 5) -> (list, list):
        """Apply SMOTE (oversampling of positive instances) by factor n."""
        data = []
        for (Xi, yi) in zip(X, y):
            if int(yi) == 1:
                for _ in range(factor):
                    data.append((Xi, yi))
            data.append((Xi, yi))
        Xo, yo = zip(*data)
        return Xo, yo

    def _cv_score(self, model: Pipeline, p_grid: dict, X: list, y: list,
                  metric: make_scorer, nest: bool = True, smote: bool = True
                  ) -> (Pipeline, float, dict):
        """Big evaluation function, handles oversampling, and cross-val."""
        neural = self.neural
        if smote:
            X, y = self._oversample(X, y, factor=3)
        if p_grid:
            # If nested add a layer of 3 splits, else just cross-validate with
            # 10. If neural apply a simple split only.
            n, _n = (10, 3) if nest else (10 if not neural else 2, 0)
            print(f"running {_n} outer, {n} inner...")
            cv = StratifiedKFold(n_splits=n, random_state=42)
            if nest:
                _cv = StratifiedKFold(n_splits=_n, random_state=42)

            # Non_nested parameter search and scoring
            grid = GridSearchCV(estimator=model, param_grid=p_grid, cv=cv,
                                scoring=metric,
                                n_jobs=1 if nest or neural else -1)
            # NOTE: n_jobs sometimes needs to be tweaked (depending on where
            # multi-threading happens). Above is the safest default config.
            grid.fit(X, y)
            print("\n> Inner CV F1:", grid.best_score_)  # Score of 10-fold

            clf = grid.best_estimator_
        else:
            try:
                assert not nest
            except AssertionError:
                raise(ValueError(
                    "Set nest to false if no p_grid is provided."))
            grid, clf = None, model

        print("\n\n> Final model:\n")
        for step in clf.steps:
            print(step)

        clf.fit(X, y)  # Refit best_estimator_ on the entire train set

        # Nested CV with parameter optimization                v only if nested
        return clf, cvs(clf, X, y, cv=_cv, scoring=metric) if nest else 0, grid

    def _cv_train(self, X_train: list, y_train: list, nest: bool = True
                  ) -> (Pipeline, ):
        """Merge params and pipe to scoring. Report nested score if needed."""
        pipeline = deepcopy(self.pipeline)  # unsure if still need (safety 1st)
        clf, s, _ = self._cv_score(
            Pipeline(list(pipeline.keys())),
            dict(ChainMap(*pipeline.values())), X_train, y_train, F1, nest)

        if nest:
            print(f"\n> Nested F1: {round(s.mean(), 3)} ({round(s.std(), 3)})")
        return clf, s

    def _store(self, data: Dataset, _data: Dataset, sets: list, clf: Pipeline
               ) -> None:
        """Log results and pickle models."""
        with open(f'./results/{data.id}__{_data.id}.txt', 'w') as fo:
            fo.write('\n'.join([f'{ŷi},{yi},"{Xi}"' for ŷi, yi, Xi in sets]))
        with open(f'./results/{data.id}__{_data.id}.pickle', 'wb') as bo:
            pickle.dump(clf, bo)

    def _top_feats(self, clf: Pipeline) -> None:
        """Report on top feature importances given SVM (or LR) classifier."""
        cv, svm = clf.steps[0][1], clf.steps[1][1]
        try:
            svm.coef_[0]
        except AttributeError:  # FIXME: fails if one nested, not robust this
            svm = clf.steps[2][1]
        topk = sorted(list(zip(cv.get_feature_names(), svm.coef_[0])),
                      key=lambda x: x[1], reverse=True)
        feat_list = [((25 - len(x[0])) * ' ').join((x[0], str(x[1])))
                     for x in topk]
        print(f"\n> Top features: \n")
        print("\n\n" + '\n'.join(feat_list[:20]) + "\n",
              "\n" + '\n'.join(feat_list[-20:]) + "\n\n")

    def score(self, datasets: list, test: list = None, nest: bool = True,
              store: bool = False, report: bool = False, df: bool = True
              ) -> None:
        """Wraps all private functionality of the class for simple syntax.

        Parameters
        ----------
        datasets: ``list``, required
            List of Dataset objects (controlled by Reader class) to be trained
            on OR a list of self-implemented classes that need an ID
            (dir + filename) and a train and test split attribute. Reader is
            recommended. :)
        test: ``list``, optional (default=None)
            Same as above, just the test instances (yes, these are duplicates).
        nest: ``bool``, optional (default=True)
            If nested cross-validation should be conducted for e.g. model
            comparison. Runs another 3-fold val over existing 10-fold.
        store: ``bool``, optional (default=False)
            Store the best model as a pickle in the /results directory. ID will
            be, again, dir + filename of data.
        report: ``bool``, optional (default=False)
            Report top best features. Requires a LR or SVM classifier (or
            anything implementing _coef that can work on get_feature_names()).
        df: ``bool``, optional (default=True)
            Dump result to dataframe (doesn't work for nl).
        """
        for data in datasets:
            print(f"\n\n> Training {data.id}")
            X_train, y_train = data.get('train')
            self._create(data.id)

            clf, _ = self._cv_train(X_train, y_train, nest)
            for _data in test:
                if not self.cross and data.id != _data.id:
                    continue
                print(f"\n\n> Testing {data.id} => {_data.id}")
                X_test, y_test = _data.get('test')

                ŷ = clf.predict(X_test)
                print(f"\n> Test results: \n")
                print(f"\n\n {sk_clf_report(y_test, ŷ, digits=3)}\n\n")
                t_score = f1_score(y_test, ŷ, average='binary', pos_label='1')

                if store:
                    self._store(data, _data, zip(ŷ, y_test, X_test), clf)
                if report:
                    self._top_feats(clf)
                if not self.merge:
                    self._add(data.id, _data.id, t_score)

        if not self.merge and self.cross and df:
            print(self.df.astype(float).round(3))
