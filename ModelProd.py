from BertModel import *
from RandomForestModel import *
from BertPreprocess import *


class ModelProd:
    def __init__(self):
        self.bert = BertModel()
        self.rf = RandomForestModel()
        self.preprocess = BertPreprocess()

    def load_pretrained(self, ):
        """
        Load previously trained models
        :return:
        """
        self.bert.load_models()
        self.rf.load_models()

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

        self.preprocess.train = pd.DataFrame({'text':[[text]],
                                              'accepted_function':[af],
                                              'rejected_function':[rf],
                                              'accepted_product':[ap],
                                              'rejected_product':[rp]})
        self.preprocess.process_text()
        emb = self.bert.model_encode(self.preprocess.train.values[0,0],
                                     self.preprocess.train.values[0,1],
                                     self.preprocess.train.values[0,2],
                                     self.preprocess.train.values[0,3],
                                     self.preprocess.train.values[0,4])
        print(emb)
        pred = self.rf.model_predict_one(emb)
        return pred
