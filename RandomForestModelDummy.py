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

class RandomForestModelDummy:
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
        pass

    def save_models(self,):
        """
        Save pretrained models in format of .pkl file for further usage
        :return:
        """
        pass

    def load_models(self, path=DATAPATH, model_file = 'rf_models.pkl'):
        """
        Loading pretrained models in format of .pkl file for further usage
        :param path: path where .pkl file is
        :param model_file: name of .pkl file
        :return:
        """
        pass

    def model_fit(self,):
        """
        Using loaded training data train two-level RandomForest model.
        :return:
        """
        pass

    def model_predict_all(self, X):
        """
        Given Dataset in form of pandas Dataframe predict result for each line.
        DataFrame should contain "id" column for correct work.
        :param X: pandas Dataframe
        :return:
        """
        return [1 for i in range(len(X))]

    def model_predict_one(self, X):
        """
        Given Dataset in form of pandas Dataframe that have only one line predict result for it
        DataFrame should NOT contain "id" column for correct work.
        :param X: pandas Dataframe
        :return:
        """
        return 1

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
