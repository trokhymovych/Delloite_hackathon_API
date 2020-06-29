import pandas as pd
import numpy as np
import pickle
import math
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import re

from BertDataReader import DataReader
from log_conf import Logger

DATAPATH = 'data/'

class BertTraining:
    def __init__(self, ):
        self.batch_size = 16
        self.reader = DataReader('')

        self.model_save_path1 = DATAPATH + 'model_dump' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model1 = SentenceTransformer('bert-base-nli-mean-tokens')

        self.model_save_path2 = DATAPATH + 'model_2_dump' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model2 = SentenceTransformer('bert-base-nli-mean-tokens')

        self.train_loss1 = losses.CosineSimilarityLoss(model=self.model1)
        self.train_loss2 = losses.CosineSimilarityLoss(model=self.model2)

        self.df1 = None
        self.train_data1 = None
        self.train_dataloader1 = None

        self.df2 = None
        self.train_data2 = None
        self.train_dataloader2 = None

    def load_data(self, data_path=DATAPATH):
        self.df1 = pd.read_csv(data_path + "product_df.csv")
        self.df1.columns = [0, 1, 2]
        Logger.logger.info("[INFO] Datasets1 for BERT training is loaded.")

        self.df2 = pd.read_csv(data_path + "function_df.csv")
        self.df2.columns = [0, 1, 2]
        Logger.logger.info("[INFO] Datasets2 for BERT training is loaded.")
        

        # training data
        self.train_data1 = SentencesDataset(self.reader.get_examples(self.df1, modelname='model_1'), model=self.model1)
        self.train_dataloader1 = DataLoader(self.train_data1, shuffle=True, batch_size=self.batch_size)
        Logger.logger.info("[INFO] First loader initialized")
        # training data
        self.train_data2 = SentencesDataset(self.reader.get_examples(self.df2, modelname='model_2'), model=self.model2)
        self.train_dataloader2 = DataLoader(self.train_data2, shuffle=True, batch_size=self.batch_size)
        Logger.logger.info("[INFO] Second loader initialized")

        # val data
        fold = int(len(self.df1) * 0.95)
        dev_data1 = SentencesDataset(self.reader.get_examples(self.df1[fold:], modelname='model_1'), model=self.model1)
        dev_dataloader1 = DataLoader(dev_data1, shuffle=False, batch_size=self.batch_size)
        self.evaluator1 = EmbeddingSimilarityEvaluator(dev_dataloader1)

        # val data
        fold = int(len(self.df2) * 0.95)
        dev_data2 = SentencesDataset(self.reader.get_examples(self.df2[fold:], modelname='model_2'), model=self.model2)
        dev_dataloader2 = DataLoader(dev_data2, shuffle=False, batch_size=self.batch_size)
        self.evaluator2 = EmbeddingSimilarityEvaluator(dev_dataloader2)

    def fit(self, ):
        # Configure the training
        num_epochs = 1
        warmup_steps = math.ceil(
            len(self.train_dataloader1) * num_epochs / self.batch_size * 0.1)  # 10% of train data for warm-up
        # Train the model
        self.model1.fit(train_objectives=[(self.train_dataloader1, self.train_loss1)],
                        evaluator=self.evaluator1,
                        epochs=num_epochs,
                        evaluation_steps=100,
                       warmup_steps=warmup_steps,
                       output_path=self.model_save_path1
                       )

        # Configure the training
        warmup_steps = math.ceil(
            len(self.train_dataloader2) * num_epochs / self.batch_size * 0.1)  # 10% of train data for warm-up

        self.model2.fit(train_objectives=[(self.train_dataloader2, self.train_loss2)],
                        evaluator=self.evaluator2,
                        epochs=num_epochs,
                        evaluation_steps=100,
                       warmup_steps=warmup_steps,
                       output_path=self.model_save_path2
                       )

    def get_embeddings(self, input_f='data/train_only_text.pkl', output_f='data/train_bert_7.pkl'):
        with open(input_f, 'rb') as f:
            train = pickle.load(f)

        train["af_embed"] = self.model2.encode(list(train["accepted_function"].values))
        train["rf_embed"] = self.model2.encode(list(train["rejected_function"].values))
        train["ap_embed"] = self.model1.encode(list(train["accepted_product"].values))
        train["rp_embed"] = self.model1.encode(list(train["rejected_product"].values))

        train["text_embed"] = self.model1.encode(list(train["text"].values))
        train["text_embed_2"] = self.model2.encode(list(train["text"].values))

        ss = train["ap_embed"][0].shape[0]

        tmp1 = pd.DataFrame().from_records(train["af_embed"])
        tmp1.columns = [f'af_{str(i)}' for i in range(ss)]

        tmp2 = pd.DataFrame().from_records(train["rf_embed"])
        tmp2.columns = [f'rf_{str(i)}' for i in range(ss)]

        tmp3 = pd.DataFrame().from_records(train["ap_embed"])
        tmp3.columns = [f'ap_{str(i)}' for i in range(ss)]

        tmp4 = pd.DataFrame().from_records(train["rp_embed"])
        tmp4.columns = [f'rp_{str(i)}' for i in range(ss)]

        tmp5 = pd.DataFrame().from_records(train["text_embed"])
        tmp5.columns = [f'tt_{str(i)}' for i in range(ss)]

        tmp6 = pd.DataFrame().from_records(train["text_embed_2"])
        tmp6.columns = [f'tt2_{str(i)}' for i in range(ss)]

        tmp = pd.concat([tmp1, tmp2, tmp3, tmp4, tmp5, tmp6], axis=1)
        tmp['id'] = train['id']
        columns1 = [f'ap_{str(i)}' for i in range(ss)] + [f'rp_{str(i)}' for i in range(ss)] + [f'tt_{str(i)}' for i in
                                                                                                range(ss)]
        columns2 = [f'af_{str(i)}' for i in range(ss)] + [f'rf_{str(i)}' for i in range(ss)] + [f'tt2_{str(i)}' for i in
                                                                                                range(ss)]

        dict_to_write = {'train_x': tmp,
                         'train_y': train.target,
                         'columns_1': columns1,
                         'columns_2': columns2
                         }

        with open(output_f, 'wb') as handle:
            pickle.dump(dict_to_write, handle, protocol=pickle.HIGHEST_PROTOCOL)
