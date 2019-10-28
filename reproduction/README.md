# Replication of Agrawal et al.

Below are my observations and code\*. The analyses were conducted on [commit `ed823d0`](https://github.com/sweta20/Detecting-Cyberbullying-Across-SMPs/commit/ed823d0b0c5ed6a563b8f17e8938690b924572d0).

> \*These are taken from my e-mail correspondence on March 7th and 14th 2019, without context nor response. Please keep this in mind.

## Observation I

What follows is my analysis of the assumed error: [This](https://github.com/sweta20/Detecting-Cyberbullying-Across-SMPs/blob/master/DNNs.ipynb) notebook was my main initial focus, which I ran locally. The error is a combination of the `get_data` function; specifically the `oversampling_rate` part, followed by the `get_train_test` function; which contains the data splitting part. As (to my understanding) shuffling and splitting of the data is not controlled for (the `train_test_split` function from `scikit-learn` shuffles by default), chances are oversampled instances bleed into the test set. This led me to further investigate how much of the positive (oversampled) instances in the train set are also in the test set (and are thus seen during training).

[See b]elow [for] the code I added to the bottom of the notebook to confirm this:

```python
import numpy as np

x_text, labels = get_data(data, oversampling_rate=3)
data_dict = get_train_test(data,  x_text, labels)

train_set = set([str(list(x)) for x in data_dict['trainX']])
test_set = set([str(list(x)) for x in data_dict['testX']])

print("overlapping instances train/test:", len(train_set & test_set))
print("nr. instances test set:", len(test_set))
print("nr. instances train set:", len(train_set))

train_pos = [x for x, y in zip(data_dict['trainX'], data_dict['trainY']) if np.argmax(y) == 1]
test_pos = [x for x, y in zip(data_dict['testX'], data_dict['testY']) if np.argmax(y) == 1]
pos_train = set([str(list(x)) for x in train_pos])
pos_test = set([str(list(x)) for x in test_pos])

print("unique test instances:", len(pos_test - pos_train))
```

> The notebook can be found in our repository under `DNNs_repl.ipynb` (older version at the time, current version was made after Observation II).

Which gives me a unique test instances number (which are not in the train set) of 1. This subsequently leads me to believe that all positive test instances are actually seen during training, and the model does not need to learn any transferable features. As far as I understand, this can also be inferred in the paper at the transfer learning experiment, where significant improvement is achieved only for the transfer methods that ignore the trained model, solely using the embeddings / weights, and thusly allowing the new model to again see the test instances during training.

## Response

> Response ommitted for privacy. Will add if requested / granted permission.

## Observation II

> Removed some additional reply context.

If we forego the featurizer part of the code, i.e. commenting out the below part from `get_train_test`, it does still give the same overlap (this time with raw documents in `trainX` and `testX`):

```python
def get_train_test(data, x_text, labels):
    
    ...
    
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    # vocab_processor = vocab_processor.fit(x_text)

    # trainX = np.array(list(vocab_processor.transform(X_train)))
    # testX = np.array(list(vocab_processor.transform(X_test)))
    
    trainY = np.asarray(Y_train)
    testY = np.asarray(Y_test)
        
    # trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    # testX = pad_sequences(testX, maxlen=max_document_length, value=0.)

    trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
    testY = to_categorical(testY, nb_classes=NUM_CLASSES)
    
    data_dict = {
        "data": data,
        "trainX" : X_train,
        "trainY" : trainY,
        "testX" : X_test,
        "testY" : testY,
        # "vocab_processor" : vocab_processor
    }
    
    return data_dict
```

(I slightly altered my added bit at the bottom):

    pos_train = set(train_pos)
    pos_test = set(test_pos)


This also allows for direct confirmation using the sets' intersection, sampling an item, and confirming if they are in the positive instances:

    len(pos_train & pos_test)


... yielding 202 items, and:

    sample_positive_instance = list(pos_train & pos_test)[0]


... taking the 0th common item, and confirming:

    sample_positive_instance in pos_train
    True


... and:

    sample_positive_instance in pos_test
    True


Therefore, I do still think my raised issue persists. I have attached my version of the DNN notebook with a few extra checks if you want to take a look at it.

> Can be found in `DNNs_repl.ipynb` (these were some changes).

As [] oversampl[ing is applied] by copying instances from the entire dataset---before splitting---those few positive instances that end up in test were also oversampled (and end up in [both train and test]). This is not fixed by disabling shuffle on the sklearn data split (I confirmed), but by only oversampling on `X_train`. I have also attached a notebook where I've adapted the code to only oversample on train, which decreases the test performance to a positive F1 score of 0.33 (granted, for the single run / set of parameters I ran). 

> The full replication correction can be found in our repository under `DNNs_oversample_train.ipynb`.

> [square brackets] are minor edits of the original e-mail.
