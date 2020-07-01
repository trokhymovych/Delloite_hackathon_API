import pandas as pd
import pickle
import string
import re

from log_conf import Logger

import spacy
from spacy.lang.en import English

DATAPATH = 'data/'

class BertPreprocess:

    def __init__(self, ):
        self.train = None
        self._stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self._parser = English()
        self._punctuations = string.punctuation

    def __del__(self, ):
        del self.train

    def load_data(self, path=DATAPATH):
        """
        Function for dataset loading. The datafile should have name
        train_df.pkl format and contain pandas DataFrame.
        :param path: string, optional. Directory, where train_df.pkl placed.
        :return:
        """
        try:
            with open(path + 'train_df.pkl', 'rb') as f:
                self.train = pickle.load(f)
            Logger.logger.info("[INFO] Data loaded")
        except Exception as e:
            Logger.logger.info("[ERROR] Problem with file loading.")
            Logger.logger.exception(e)
            pass

    def process_text(self, ):
        """
        Process provided dataset. Dataframe should contain "text" field.
        :return:
        """
        if self.train is not None:
            self.train["clear_text"] = self.train["text"].apply(self._make_list_of_texts)
            self.train["text"] = self.train.clear_text.apply(self._combine_sentences)
            self.train = self.train[["text", "accepted_function", "rejected_function",
                                     "accepted_product", "rejected_product",'target']]
            for col in self.train.columns[:5]:
                self.train[col] = self.train[col].apply(self._replace_nan)
            Logger.logger.info("[INFO] The trainset is processed.")
        else:
            Logger.logger.info("[ERROR] The trainset is not uploaded. Use 'load' method.")


    def create_datasets_for_bert(self, path=DATAPATH):
        """
        Create datasets for BERT finetuning in special format
        :param path:
        :return:
        """
        if self.train is not None:
            tmp_product = self._format_dataset("accepted_product", "rejected_product")
            tmp_product.to_csv(path + "product_df.csv", index=False)

            tmp_function = self._format_dataset("accepted_function", "rejected_function")
            tmp_function.to_csv(path + "function_df.csv", index=False)
            Logger.logger.info("[INFO] The trainsets for BERT models are prepared.")
        else:
            Logger.logger.info("[ERROR] The trainset is not uploaded. Use 'load' method.")

    def save_result(self,):
        """
        Save raw preprocesed dataframe to train_only_text.pkl file
        :param name_of_file: string
        :return:
        """
        if self.train is not None:
            with open('data/train_only_text.pkl', 'wb') as f:
                pickle.dump(self.train, f)
            Logger.logger.info("[INFO] File is saved.")
        else:
            Logger.logger.info("[INFO] Nothing to save.")

    def _format_dataset(self, accepted_column, rejected_column):
        """
        Create dataset with given criteria.
        :param accepted_column: string
        :param rejected_column: string
        :return:
        """
        fin_columns = ["text", "criteria", "score"]

        final_df = pd.DataFrame(columns=fin_columns)
        sub = self.train[["text", accepted_column, "target"]]
        sub.columns = fin_columns
        sub = sub[sub.criteria.str.len() > 0]
        sub = sub[sub.text.str.len() > 0]
        sub.score = sub.score.apply(lambda x: x == 2).astype("int")
        final_df = final_df.append(sub)

        sub = self.train[["text", rejected_column, "target"]]
        sub.columns = fin_columns
        sub = sub[sub.criteria.str.len() > 0]
        sub = sub[sub.text.str.len() > 0]
        sub.score = sub.score.apply(lambda x: x < 2).astype("int")
        final_df = final_df.append(sub)
        return final_df

    def _make_list_of_texts(self, texts):
        """
        Function to preprocess and change text to appropriate for model format
        :param texts: list of strings
        :return: string
        """
        texts = '\n'.join(texts)
        texts = re.sub("\.|\!", "\n", texts)
        sentences = [self._spacy_tokenizer(self._clear_text(sen)) for sen in texts.split('\n')]
        return [' '.join(sen) for sen in sentences if len(sen) > 3]

    def _spacy_tokenizer(self, sentence):
        """
        String tokenizer, using standard spaCy parser and stopwords
        :param sentence: string
        :return: list of strings
        """
        mytokens = self._parser(sentence)
        mytokens = [word.lower_ for word in mytokens]
        mytokens = [word for word in mytokens if
                    word not in self._stop_words and word not in self._punctuations and not word.isdigit() and len(
                        word) > 2]
        return mytokens

    @staticmethod
    def _clear_text(text):
        """
        Lowercase the string and remove unappropriated signs.
        :param text: string
        :return: string
        """
        text = text.lower()
        text = re.sub("\n|\r|\t", " ", text)
        text = re.sub("[^\w\s\.]", " ", text)
        text = re.sub(" +", " ", text.strip())
        return text

    @staticmethod
    def _combine_sentences(list_of_sent):
        """
        Making one string out on list of strings joining strings with "."
        :param list_of_sent: list of strings
        :return: string
        """
        result = " "
        for sent in list_of_sent:
            result += sent + ". "
        return result.strip()

    @staticmethod
    def _replace_nan(text):
        """
        Replace NaN fields with special _nan_ token
        :param text: string
        :return:
        """
        return re.sub("_nan_", "", text)
