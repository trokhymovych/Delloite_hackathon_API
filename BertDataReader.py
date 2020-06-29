from sentence_transformers.readers import *
import csv


##### Custom reader
class DataReader:
    """
    Reads in the STS dataset. Each line contains two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)
    """
    def __init__(self, dataset_folder, s1_col_idx=0, s2_col_idx=1, score_col_idx=2, delimiter="\t",
                 quoting=csv.QUOTE_NONE, normalize_scores=False, min_score=0, max_score=1):
        self.dataset_folder = dataset_folder
        self.score_col_idx = score_col_idx
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score

    def get_examples(self, df, modelname = 'model_1', max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        examples = []
        for r in df.iterrows():
            id = r[0]
            row = r[1]
            score = float(row[self.score_col_idx])

            s1 = row[self.s1_col_idx]
            s2 = row[self.s2_col_idx]
            examples.append(InputExample(guid=modelname+str(id), texts=[s1, s2], label=score))

        return examples