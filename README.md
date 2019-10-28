# Current Limitations in Cyberbullying Detection: on Evaluation Criteria, Reproducibility, and Data Scarcity

Repository for the work described in [Current Limitations in Cyberbullying Detection: on Evaluation Criteria, Reproducibility, and Data Scarcity](#). Code is released under the GPL-v3 license. If you use anything related to the repository or paper, please cite the following work:

```
@article{emmery2019current,
  title={Current Limitations in Cyberbullying Detection: on Evaluation
         Criteria, Reproducibility, and Data Scarcity},
  author={Emmery, Chris and Verhoeven, Ben and De Pauw, Guy and Jacobs, Gilles
          and Van Hee, Cynthia and Lefever, Els and Desmet, Bart and Hoste,
          V\'{e}ronique and Daelemans, Walter},
  journal={arXiv preprint arXiv:EDIT ON ANNOUNCED},
  year={2019}
}
```

> **Article not announced yet!**

Also consider citing the corpora that we used to compare when using the readers supplied with our work. Refer to our paper for references.

## Overview

- [tl;dr](https://github.com/cmry/amica#tldr)
- [Quick Start](https://github.com/cmry/amica#quick-start)
  - [Reproduction](https://github.com/cmry/amica#reproduction)
  - [Test your Own Pipeline](https://github.com/cmry/amica#test-your-own-pipeline)
  - [Including New Data](https://github.com/cmry/amica#including-new-data)
  - [Debugging](https://github.com/cmry/amica#debugging)
- [Dependencies](https://github.com/cmry/amica#dependencies)

## tl;dr

This repository offers scripts to compare (scikit-learn-API-based) classifiers on various cyberbullying detection corpora, including suggestions for different data augmentation methods, and easy cross-domain evaluation. We showed this provides more detailed insight into the limitations of such classifiers, and allows for stronger (more critical) comparisons between them.


## Quick Start

Not only do we supply code to *replicate* our experiments, we also offer the API to subject new models to the same evaluation.

> **Support Disclaimer:** The code is written on a Linux system, using Python 3.7. Some functionality might not be portable, please check [Debugging](https://github.com/cmry/amica#debugging), and the code's `#NOTE` comments if certain things do not work.

### Reproduction

All default parameter settings are according to the paper. Please check the help (`python experiments.py -h`) for details). Example use, to replicate Table 4:

```shell
python experiments.py baseline
python experiments.py baseline --merge
```
> **Note on Score Reproduction**: As long as the paper is pre-print, we might still change the experiments. Please make sure to refer to the most recent paper version (will be updated in this repository).

The reproduction of the neural models from Agrawal et al., including extensive documentation, can be found under `/reproduction`.

#### Data

We do *not* supply the data with this repository. We included most of the required scripts, pointers, etc. for the open-source corpora in the `/corpora` directory. If you are a researcher interested in *replication*, please contact us for the data (contact info in paper).

The readers in our repository assume that all data is in a `.csv` format with `label,"text of a document"` as columns.

If you want to run without certain corpora, either comment out the tuples in [this]() part. Alternatively, if you'd like to add your own data for the comparison, please see the following sections.


### Debugging

We provide a debugging script testing all current functionality of the main evaluation pipeline under `utils.py` on a small debugging dataset (found under `/corpora`). It can be run from shell like so:

```shell
python experiments.py debug

testing preprocessing ... ok
testing merge . ok
testing single_domain . ok
testing multi_thread . ok
testing store . ok
testing report . ok
testing model ....... ok

... Test was a success, congrats!
```

Depending on versions of packages used, this might throw a few `Deprecation` and `FutureWarnings` in between. Please see [Dependencies](https://github.com/cmry/amica#dependencies) first if anything fails.

### Test your Own Pipeline

Adapting the `scikit-learn` API to implement a custom pipline into the
framework is fairly straight-forward. Consider the following code for generating
the Naive Bayes features used in NB-SVM:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class BayesFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.r = None
    
    def pr(self, X, y_i, y):
        p = X[[int(yi == y_i) for yi in y]].sum(0)
        return (p + 1) / (sum([int(yi) == y_i for yi in y]) + 1)
        
    def fit(self, X, y):
        self.r = np.log(self.pr(X, 1, y) / self.pr(X, 0, y))
        return self
    
    def transform(self, X):
        return X.multiply(self.r)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
```

The only requirements are adapting the base API for estimators and transformer
modules by inheritance (`(BaseEstimator, TransformerMixin)` in the `class`
definition. Just supply `fit`, `transform`, and `fit_transform` functions,
with matching parameters (`X`, optionally `y`), and make sure to `return self`
in `fit`.

Classifiers are implemented in a similar way:

```python
from sklearn.base import BaseEstimator, ClassifierMixin

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
```

Note that rather than `transform`, there is now a `predict` method that returns labels (i.e., `ŷ`), and method inheritance from `ClassifierMixin` is required.

Alternatively, you may use whatever sklearn already provides. Anything adhering to their API can be included as a `pipeline` to the experiments like so:

```python
your_pipeline = {
    ('vect', CountVectorizer(binary=True)): {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)]
    },
    ('svc', LinearSVC(random_state=42)): {
        'svc__C': [1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3],
        'svc__loss': ['hinge', 'squared_hinge'],
        'svc__class_weight': [None, "balanced"]
    }
}
```

Note that keys are tuples with `(id name` (a string)`, sklearn-API class` (an object)`)`, and values another dictionary, with `{parameter name `(a string)`: [list with parameter values` (also in strings)`]}`. This is formatted according to sklearn's [`Pipeline` class](https://scikit-learn.org/stable/modules/compose.html#pipeline) (just with some additional nesting).

You can provide this (and other arguments) to the experiments like so:

```python
Experiment(pipeline=your_pipeline).run()
```

If you want to also include new data here, read on.

### Including New Data

The most hassle-free way of including your own data is copying (or linking) it to the `/corpora` directory. The current structure should be as follows:

```shell
corpora
├── yourname
│   ├── some_directory_name
│   │   ├── dataname.csv
```

In the current implementation, the data needs to be nested in a directory deeper than one would expect. It's somewhat counter-intuitive, more so as the data tuple ignores this name. It should be referred to as `('yourname', 'dataname')` in the current example.

However, in this way you can simply run your own pipeline and include the tuple in the experiment parameters, like so:

```python
Experiment(pipeline=your_pipeline, datasets=[('yourname', 'dataname']).run()
```

For additional documentation, please refer to the docstrings of the classes. They are fairly detailed.

## Dependencies

These are the required (non-standard) packages (and their version tested with the latest version of the repository) to run the **full** repository:

```python
# base
scikit-learn==0.21.3
pandas==0.25.2
emot==2.1
spacy==2.2.1
tqdm==4.36.1
numpy=1.16.4

# optional
keras==2.3.1            # (for reproducing neural models)
tensorflow==1.13.1      # same, backend to keras
tflearn==0.3.2          # for padding etc. in reproduction
transformers==2.1.1     # for DistilBERT
torch==1.3.0            # for transformers
gensim==3.8.1           # for word2vec
```
