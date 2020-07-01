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

DATAPATH = 'data/'
RANDOM_SEED = 42

class RandomForestModel:
    def __init__(self,):
        self.model1 = None
        self.model2 = None
        self.columns_1 = []
        self.columns_2 = []
        self.X = None
        self.y = None

    def load_data(self, filename = 'train_bert_7.pkl'):
        """
        Loading training data in format on pickle file. The file
        containing stacked embeddings from bert in form of pandas DataFrame
        :param filename: filename
        :return:
        """
        with open(DATAPATH+filename, 'rb') as handle:
            train_bert = pickle.load(handle)

        self.X = train_bert['train_x']
        self.y = train_bert['train_y']

        self.columns_1 = train_bert['columns_1']
        self.columns_2 = train_bert['columns_2']

    def save_models(self,):
        """
        Save pretrained models in format of .pkl file for further usage
        :return:
        """
        dict_to_save = {'m1':self.model1, 'm2':self.model2, 'c1':self.columns_1, 'c2':self.columns_2}
        with open(DATAPATH+'rf_models.pkl', 'wb') as handle:
            pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_models(self, path=DATAPATH, model_file = 'rf_models.pkl'):
        """
        Loading pretrained models in format of .pkl file for further usage
        :param path: path where .pkl file is
        :param model_file: name of .pkl file
        :return:
        """
        with open(path+model_file, 'rb') as handle:
            dict_to_load = pickle.load(handle)
        self.model1 = dict_to_load['m1']
        self.model2 = dict_to_load['m2']
        self.columns_1 = dict_to_load['c1']
        self.columns_2 = dict_to_load['c2']

    def model_fit(self,):
        """
        Using loaded training data train two-level RandomForest model.
        :return:
        """
        # first level model
        y_1 = self.y.apply(self._first_level_target)
        X_1 = self.X[self.columns_1]

        clf1 = RandomForestClassifier(n_estimators=200,
                                      min_samples_split=10,
                                      min_samples_leaf=2,
                                      max_features='sqrt',
                                      max_depth=50,
                                      bootstrap =True,
                                      n_jobs = 4)
        clf1 = clf1.fit(X_1, y_1)

        # second level model
        y_2 = self.y[self.y.apply(self._first_level_target).astype(bool)]
        X_2 = self.X[self.y.apply(self._first_level_target).astype(bool)][self.columns_2]

        clf2 = RandomForestClassifier(n_estimators=200,
                                      min_samples_split=10,
                                      min_samples_leaf=2,
                                      max_features='sqrt',
                                      max_depth=50,
                                      bootstrap =True,
                                      n_jobs = 4,
                                      class_weight = 'balanced')
        clf2 = clf2.fit(X_2, y_2)

        self.model1 = clf1
        self.model2 = clf2


    def model_predict_all(self, X):
        """
        Given Dataset in form of pandas Dataframe predict result for each line.
        DataFrame should contain "id" column for correct work.
        :param X: pandas Dataframe
        :return:
        """
        # first level model
        X_1 = X[self.columns_1]

        preds1 = self.model1.predict(X_1)
        zero_ids = X[~preds1.astype(bool)].id

        res0 = pd.DataFrame({'id': zero_ids, 'target': [0 for i in range(len(zero_ids))]})

        # second level model
        X_2 = X[preds1.astype(bool)][self.columns_2]

        preds2 = self.model2.predict(X_2)

        ones_ids = X[preds1.astype(bool)][preds2 == 1].id

        res1 = pd.DataFrame({'id': ones_ids, 'target': [1 for i in range(len(ones_ids))]})

        two_ids = X[preds1.astype(bool)][preds2 == 2].id
        res2 = pd.DataFrame({'id': two_ids, 'target': [2 for i in range(len(two_ids))]})

        res = pd.concat([res0, res1, res2], axis=0)
        res.index = res.id

        return res.loc[X.id.values, 'target'].values

    def model_predict_one(self, X):
        """
        Given Dataset in form of pandas Dataframe that have only one line predict result for it
        DataFrame should NOT contain "id" column for correct work.
        :param X: pandas Dataframe
        :return:
        """
        # first level model
        X_1 = X[self.columns_1]

        pred = self.model1.predict(X_1)

        if pred[0] == 0:
            return 0

        else:
            # second level model
            X_2 = X[self.columns_2]

            pred2 = self.model2.predict(X_2)

            if pred2 == 1:
                return 1
            else:
                return 2

    @staticmethod
    def _weighted_accuracy(y_true, y_pred):
        weights = [2 if y == 2 else 1 for y in y_true]
        return np.sum(weights * (y_true == y_pred)) / sum(weights)

    @staticmethod
    def _set_seed(random_state=RANDOM_SEED):
        random.seed(random_state)
        np.random.seed(random_state)

    @staticmethod
    def _first_level_target(x):
        if x == 0:
            return 0
        else:
            return 1
