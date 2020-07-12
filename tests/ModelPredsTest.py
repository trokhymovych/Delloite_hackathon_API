from __future__ import print_function
import sys
import os
sys.path.insert(1, os.getcwd()+'/../modules')

from BertPreprocess import *
from ModelProd import *

import unittest
import numpy as np


class ModelPredsTest(unittest.TestCase):
    def setUp(self):
        self.data_model = BertPreprocess()
        self.data_model.load_data()
        self.testset = self.data_model.train.head(5)
        self.model = ModelProd()
        self.model.load_pretrained()

    def test_smoke_test_model(self):
        res = []
        for row_id, row in self.testset.iterrows():
            i = self.model.predict('\n'.join(row.text),
                                   row['accepted_function'],
                                   row['rejected_function'],
                                   row['accepted_product'],
                                   row['rejected_product'])
            res.append(i)
        print(res)
        self.assertFalse(0 in res), "Should contain at least on 0 in prediction"
        self.assertTrue(1 in res), "Should contain at least on 1 in prediction"
        self.assertTrue(2 in res), "Should contain at least on 2 in prediction"


if __name__ == "__main__":
    unittest.main()
