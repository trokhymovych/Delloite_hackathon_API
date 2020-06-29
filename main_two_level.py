import os
import sys
import logging
import random
import pickle

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd

DATA_PATH = 'data'
RANDOM_SEED = 42
DUMP_PATH = os.path.join(DATA_PATH, 'data.pkl')
USE_DUMPED_DATA = True
USE_SCALER = True


def weighted_accuracy(y_true, y_pred):
    weights = [2 if y == 2 else 1 for y in y_true]
    return np.sum(weights * (y_true == y_pred)) / sum(weights)


def set_seed(random_state=RANDOM_SEED):
    random.seed(random_state)
    np.random.seed(random_state)


def first_level_target(x):
    if x == 0:
        return 0
    else:
        return 1


def model_fit(X, y, columns_1, columns_2):

    # first level model
    y_1 = y.apply(first_level_target)
    X_1 = X[columns_1]

    clf1 = RandomForestClassifier(n_estimators=200,
                                  min_samples_split=10,
                                  min_samples_leaf=2,
                                  max_features='sqrt',
                                  max_depth=50,
                                  bootstrap =True,
                                  n_jobs = 4)
    clf1 = clf1.fit(X_1, y_1)

    # second level model
    y_2 = y[y.apply(first_level_target).astype(bool)]
    X_2 = X[y.apply(first_level_target).astype(bool)][columns_2]

    clf2 = RandomForestClassifier(n_estimators=200,
                                  min_samples_split=10,
                                  min_samples_leaf=2,
                                  max_features='sqrt',
                                  max_depth=50,
                                  bootstrap =True,
                                  n_jobs = 4,
                                  class_weight = 'balanced')
    clf2 = clf2.fit(X_2, y_2)

    return (clf1, clf2)


def model_predict(X, models, columns_1, columns_2):
    # first level model
    X_1 = X[columns_1]

    preds1 = models[0].predict(X_1)
    zero_ids = X[~preds1.astype(bool)].id

    res0 = pd.DataFrame({'id': zero_ids, 'target': [0 for i in range(len(zero_ids))]})

    # second level model
    X_2 = X[preds1.astype(bool)][columns_2]

    preds2 = models[1].predict(X_2)

    ones_ids = X[preds1.astype(bool)][preds2 == 1].id
    res1 = pd.DataFrame({'id': ones_ids, 'target': [1 for i in range(len(ones_ids))]})

    two_ids = X[preds1.astype(bool)][preds2 == 2].id
    res2 = pd.DataFrame({'id': two_ids, 'target': [2 for i in range(len(two_ids))]})

    res = pd.concat([res0, res1, res2], axis=0)
    res.index = res.id

    return res.loc[X.id.values, 'target'].values


if __name__ == '__main__':
    # if you face troubles with importing spacy vocab: exec "python -m spacy download en"
    # TODO:: implement train scaler (min-max or mean-sd) to normalize data before fit
    # TODO:: consider using dim.reduction (PCA?) for BoW features

    set_seed()

    logging.basicConfig(filename="logs.log", filemode='w', level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('Processing data')

    ######
    with open('data/train_bert_7.pickle', 'rb') as handle:
        train_bert = pickle.load(handle)

    with open('data/test_bert_7.pickle', 'rb') as handle:
        test_bert = pickle.load(handle)
    ######

    logging.info('Processing data finished')

    X = train_bert['train_x']
    y = train_bert['train_y']
    X_test = test_bert['test_x']

    columns_1 = train_bert['columns_1']
    columns_2 = train_bert['columns_2']


    logging.info('Splitting to CV splits')
    kf = KFold(n_splits=4, random_state=RANDOM_SEED, shuffle=True)
    kf.get_n_splits(X)

    mm_scaler = preprocessing.MaxAbsScaler()

    waccs = []
    fold = 1
    for train_index, test_index in kf.split(X):

        logging.info('FOLD #{}'.format(fold))
        fold += 1
        X_train, X_val = X.loc[train_index], X.loc[test_index]
        y_train, y_val = y.loc[train_index], y.loc[test_index]

        if USE_SCALER:
            cols = X_train.columns[:-1]
            X_train[cols] = mm_scaler.fit_transform(X_train[cols])
            X_val[cols] = mm_scaler.transform(X_val[cols])


        logging.info('START OF MODEL FIT')
        models = model_fit(X_train, y_train, columns_1, columns_2)
        logging.info('END OF MODEL FIT')

        pred_val = model_predict(X_val, models, columns_1, columns_2)

        w_acc = weighted_accuracy(y_val, pred_val)
        logging.info('Validation accuracy: {}'.format(metrics.accuracy_score(y_val, pred_val)))
        logging.info('Validation waccuracy: {}'.format(w_acc))
        waccs.append(w_acc)


    logging.info('Fitting final model on whole dataset')

    if USE_SCALER:
        cols = X.columns[:-1]
        X[cols] = mm_scaler.fit_transform(X[cols])
        X_test[cols] = mm_scaler.transform(X_test[cols])

    models = model_fit(X, y, columns_1, columns_2)

    mean_wacc = np.mean(waccs)
    logging.info('MEAN OF weighted accuracies: {}'.format(mean_wacc))

    logging.info('Predicting test set values')
    pred_test = model_predict(X_test, models, columns_1, columns_2)

    #### submition
    sample = pd.read_csv("data/sample_submission.csv")

    submission_df = pd.DataFrame(
        {'id': sample['id'], 'target': pred_test.astype(str)}
    )
    submission_df.to_csv("all_submissions/submission_Mykola_1.csv", index=False)
