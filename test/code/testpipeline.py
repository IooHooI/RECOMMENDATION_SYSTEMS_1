import unittest

import pandas as pd

from sklearn.pipeline import Pipeline

from source.code.sparse import SparseMatrixCreator
from source.code.cfsvd import SVDRecommender
from source.code.split import TrainTestSplitter
from source.code.utils import preprocessing


class TestPipeline(unittest.TestCase):

    def setUp(self):
        enc = 'Windows-1251'
        addr = '../../notebooks/data/BX-{}.csv'
        self.ratings = pd.read_csv(addr.format('Book-Ratings'), sep=';', header=0, error_bad_lines=False, encoding=enc)
        self.books = pd.read_csv(addr.format('Books'), sep=';', header=0, error_bad_lines=False, encoding=enc)
        self.users = pd.read_csv(addr.format('Users'), sep=';', header=0, error_bad_lines=False, encoding=enc)
        self.data_dict = {'books': self.books, 'users': self.users, 'ratings': self.ratings}

    def test_case_1(self):
        pipeline = Pipeline([
            ('sparse', SparseMatrixCreator()),
            ('fit', SVDRecommender(n_components=2))
        ])

        preprecessed_data_dict = preprocessing(self.data_dict, 50, 50)

        tds = TrainTestSplitter(preprecessed_data_dict, 100, 0.2)

        train, test = next(tds.__iter__())

        pipeline.fit(train)
        pipeline._final_estimator.predict(test)

    def test_case_2(self):
        pipeline = Pipeline([
            ('sparse', SparseMatrixCreator()),
            ('fit', SVDRecommender(n_components=2))
        ])

        pipeline.fit(preprocessing(self.data_dict, 50, 50))

    def test_case_3(self):
        pipeline = Pipeline([
            ('sparse', SparseMatrixCreator()),
            ('fit', SVDRecommender(n_components=2))
        ])

        pipeline.fit(preprocessing(self.data_dict, 50, 50))


if __name__ == '__main__':
    unittest.main()
