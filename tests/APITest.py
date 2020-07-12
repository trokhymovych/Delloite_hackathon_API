from __future__ import print_function

import urllib.request
import json

import unittest
import numpy as np


class ModelPredsTest(unittest.TestCase):

    def test_smoke_test_API(self):
        query = """
        http://0.0.0.0:8001/model/?company_page=https%3A%2F%2Fwww.ciklum.com%2F&rejected_product=Automotive%2C%20insurance%2C%20gambling%2C%20dating%2C%20sport&accepted_product=IT%2C%20AI%2C%20machine%20learning%2C%20finance
        """
        response = urllib.request.urlopen(query)
        res = json.loads(response.read())

        self.assertTrue(res["Answer"]== "Rejected by product"), "Incorrect result"

if __name__ == "__main__":
    unittest.main()