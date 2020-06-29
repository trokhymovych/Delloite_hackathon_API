import os
import sys
import logging
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

from feature_extraction.dataset_reader import DatasetProcessor

DATA_PATH = 'data'
RANDOM_SEED = 42
DUMP_PATH = os.path.join(DATA_PATH, 'data.pkl')
USE_DUMPED_DATA = True


def weighted_accuracy(y_true, y_pred):
    weights = [2 if y == 2 else 1 for y in y_true]
    return np.sum(weights * (y_true == y_pred)) / sum(weights)


def set_seed(random_state=RANDOM_SEED):
    random.seed(random_state)
    np.random.seed(random_state)


if __name__ == '__main__':
    # if you face troubles with importing spacy vocab: exec "python -m spacy download en"
    # TODO:: implement train scaler (min-max or mean-sd) to normalize data before fit
    # TODO:: consider using dim.reduction (PCA?) for BoW features

    set_seed()

    logging.basicConfig(filename="logs.log", filemode='w', level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('Processing data')
    data = DatasetProcessor(train_path=os.path.join(DATA_PATH, 'train_df.pkl'),
                            test_path=os.path.join(DATA_PATH, 'test_df.pkl'),
                            validate_pipe=False)

    if USE_DUMPED_DATA and os.path.exists(DUMP_PATH):
        logging.info('Will use dumped data')
        with open(DUMP_PATH, 'rb') as f:
            data = pickle.load(f)
    else:
        data.get_vectorization()

        with open(DUMP_PATH, 'wb') as f:
            pickle.dump(data, f)

    X = data.train_x
    y = data.train_y
    X_test = data.test_x

    logging.info('Splitting to CV splits')
    kf = KFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
    kf.get_n_splits(X)
    waccs = []
    fold = 1
    for train_index, test_index in kf.split(X):
        logging.info('FOLD #{}'.format(fold))
        fold += 1
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        logging.info('START OF MODEL FIT')
        classifier = LogisticRegression(max_iter=5000)
        classifier.fit(X_train, y_train)
        logging.info('END OF MODEL FIT')

        pred_val = classifier.predict(X_val)
        w_acc = weighted_accuracy(y_val, pred_val)
        logging.info('Validation accuracy: {}'.format(metrics.accuracy_score(y_val, pred_val)))
        logging.info('Validation waccuracy: {}'.format(w_acc))
        waccs.append(w_acc)

    logging.info('Fitting final model on whole dataset')
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, y)
    mean_wacc = np.mean(waccs)
    logging.info('MEAN OF weighted accuracies: {}'.format(mean_wacc))

    logging.info('Predicting test set values')
    pred_test = classifier.predict(X_test)
    data.get_submission_csv(pred_test, mean_wacc)
