from __future__ import annotations

import os
import sys
from copy import deepcopy
from argparse import Namespace

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.classification import _check_targets
from sklearn.utils.multiclass import unique_labels
import numpy as np
import warnings


def sk_clf_report(y_true: list, y_pred: list, labels: list = None,
                  target_names: list = None, sample_weight: list = None,
                  digits: int = 2, output_dict: bool = False) -> str:
    """Build a text report showing the main classification metrics

    Read more in the :ref:`User Guide <classification_report>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.
    target_names : list of strings
        Optional display names matching the labels (same order).
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    digits : int
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.
    output_dict : bool (default = False)
        If True, return output as dict

    Returns
    -------
    report : string / dict
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }
        The reported averages include micro average (averaging the
        total true positives, false negatives and false positives), macro
        average (averaging the unweighted mean per label), weighted average
        (averaging the support-weighted mean per label) and sample average
        (only for multilabel classification). See also
        :func:`precision_recall_fscore_support` for more details on averages.
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".

    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                  precision    recall  f1-score   support
    <BLANKLINE>
         class 0       0.50      1.00      0.67         1
         class 1       0.00      0.00      0.00         1
         class 2       1.00      0.67      0.80         3
    <BLANKLINE>
       micro avg       0.60      0.60      0.60         5
       macro avg       0.50      0.56      0.49         5
    weighted avg       0.70      0.60      0.61         5
    <BLANKLINE>
    """

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    labels_given = True
    if labels is None:
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        labels = np.asarray(labels)

    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn(
                "labels size, {0}, does not match size of target_names, {1}"
                .format(len(labels), len(target_names))
            )
        else:
            raise ValueError(
                "Number of classes, {0}, does not match size of "
                "target_names, {1}. Try specifying the labels "
                "parameter".format(len(labels), len(target_names))
            )
    if target_names is None:
        target_names = [u'%s' % l for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)
    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith('multilabel'):
        average_options = ('micro', 'macro', 'weighted', 'samples')
    else:
        average_options = ('micro', 'macro', 'weighted')

    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers,
                                          [i.item() for i in scores]))
    else:
        longest_last_line_heading = 'weighted avg'
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
        report = head_fmt.format(u'', *headers, width=width)
        report += u'\n\n'
        row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += u'\n'

    # compute all applicable averages
    for average in average_options:
        line_heading = average + ' avg'
        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels,
            average=average, sample_weight=sample_weight)
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]

        if output_dict:
            report_dict[line_heading] = dict(
                zip(headers, [i.item() for i in avg]))
        else:
            report += row_fmt.format(line_heading, *avg,
                                     width=width, digits=digits)

    if output_dict:
        return report_dict
    else:
        return report


def block_printing(func):
    """Wrapper to block printing stuff during testing."""
    def func_wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        value = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return value
    return func_wrapper


@block_printing
def call_experiment(args: Namespace, exp: object, select_model: object
                    ) -> None:
    """Wrapper to call experiment on debug set given arguments."""
    datasets = [('debug', 'debug_set'), ('debug', 'debug2_set')]
    exp(pipeline=select_model(args.model), merge=args.merge,
        datasets=datasets, cross=args.single_domain,
        neural=args.model in ['blstm', 'cnn', 'nn'],
        clean='clean' in args.preprocessing,
        preprocess='preprocess' in args.preprocessing,
        multi_read=args.multi_read).run(nest=args.nest,
                                        store=args.store,
                                        report=args.report)


def test_report(name: str, parameters: list, args: Namespace,
                exp: object, select_model: object) -> None:
    """Report on changed arguments test."""
    reset_args = deepcopy(args)
    print(f"testing {name}", end=' ', flush=True)
    for param in parameters:
        print(".", end='', flush=True)
        args.__dict__[f"{name}"] = param
        call_experiment(args, exp, select_model)
        args.__dict__ = reset_args.__dict__
    print(" ok")


def debug_tests(*args) -> None:
    """Test all possible inputs for args."""
    try:
        print("Running debugger: \n")
        # NOTE: smallest sets for debugging, works for English only
        if args[0].language != 'en':
            raise(ValueError, "Debugger only works for English!")

        test_report('preprocessing', ['none', 'clean', 'preprocess'], *args)
        test_report('merge', [True], *args)
        # NOTE: test_report('nest', [True], *args) -- only on grid
        test_report('single_domain', [True], *args)
        test_report('multi_thread', [2], *args)
        test_report('store', [True], *args)
        test_report('report', [True], *args)

        test_report('model', [f"debug-{x}" for x in
                              ['baseline', 'nbsvm', 'w2v', 'bert', 'blstm',
                               'cnn', 'nn']], *args)

        print("\n... Test was a success, congrats!")
    except Exception as e:
        print("\n... Test failed, error: \n")
        raise e
