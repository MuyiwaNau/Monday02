import unittest

from sklearn.linear_model import LogisticRegression

import muyi


class TestMuyi(unittest.TestCase):

    def test_model_type(self):
        model = muyi.model
        self.assertIsInstance(model, LogisticRegression)


if __name__ == '__main__':
    unittest.main()
