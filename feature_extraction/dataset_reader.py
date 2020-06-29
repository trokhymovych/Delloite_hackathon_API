import os

import pandas as pd

import feature_extraction.text_tokenizer as fe


def read_train_test_df(data_dir, train_name='train_df.pkl', test_name='test_df.pkl'):
    train = pd.read_pickle(os.path.join(data_dir, train_name))
    test = pd.read_pickle(os.path.join(data_dir, test_name))
    return train, test


class DatasetProcessor:
    def __init__(self, train_path, test_path, validate_pipe=False):
        self.train_path = train_path
        self.test_path = test_path
        self.validate_pipe = validate_pipe
        self.train = None
        self.test = None
        self.train_x = None
        self.train_y = None
        self.test_x = None

    def read_datasets(self):
        self.train = pd.read_pickle(self.train_path)
        self.test = pd.read_pickle(self.test_path)
        if self.validate_pipe:
            self.train = self.train.head(200)
            self.test = self.test.head(150)

    def clear_texts(self):
        if self.train is None or self.test is None:
            raise RuntimeError('Datasets were not read yet. Use read_datasets().')

        self.train["clear_text"] = self.train["text"].apply(fe.make_one_text)
        self.test["clear_text"] = self.test["text"].apply(fe.make_one_text)

    def vectorize(self, vectorizer=fe.bow_vector):
        vectorized = vectorizer.fit_transform(list(self.train['clear_text']) + list(self.test['clear_text']))
        self.train_x = vectorized[:len(self.train)]
        self.test_x = vectorized[len(self.train):]

    def get_vectorization(self):
        self.read_datasets()
        self.clear_texts()
        self.vectorize()
        self.train_y = self.train["target"]

    def get_submission_csv(self, pred_labels, score_achived=0):
        submission_df = pd.DataFrame(
            {'id': self.test['id'], 'target': pred_labels}
        )
        submission_df.to_csv('submission.csv', index=False)
        with open('score.txt', 'w') as f:
            f.write(str(int(score_achived * 1e+5)))
