from BertModel import *
from RandomForestModel import *
from BertPreprocess import *


class ModelProdDummy:
    def __init__(self):
        self.bert = BertModel()
        self.rf = RandomForestModel()
        self.preprocess = BertPreprocess()

    def load_pretrained(self, ):
        """
        Load previously trained models
        :return:
        """
        pass

    def predict(self, text, af, rf, ap, rp):
        """
        Predict the final output
        :param text: string
        :param af: string
        :param rf: string
        :param ap: string
        :param rp: string
        :return:
        """
        return 1
