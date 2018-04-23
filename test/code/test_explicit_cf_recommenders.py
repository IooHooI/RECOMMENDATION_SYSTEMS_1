import unittest

import pandas as pd

from sklearn.pipeline import Pipeline

from source.code.sparse import SparseMatrixCreator
from source.code.cfsvd import SVDRecommender
from source.code.cfnmf import NMFRecommender
from source.code.split import TrainTestSplitter
from source.code.utils import preprocessing
from source.code.metrics import rmse


class TestSVDPipeline(unittest.TestCase):

    def setUp(self):
        enc = 'Windows-1251'
        addr = '../../notebooks/data/BX-{}.csv'
        self.ratings = pd.read_csv(
            filepath_or_buffer=addr.format('Book-Ratings'),
            error_bad_lines=False,
            warn_bad_lines=False,
            low_memory=False,
            encoding=enc,
            header=0,
            sep=';'
        )
        self.books = pd.read_csv(
            filepath_or_buffer=addr.format('Books'),
            error_bad_lines=False,
            warn_bad_lines=False,
            low_memory=False,
            encoding=enc,
            header=0,
            sep=';'
        )
        self.users = pd.read_csv(
            filepath_or_buffer=addr.format('Users'),
            error_bad_lines=False,
            warn_bad_lines=False,
            low_memory=False,
            encoding=enc,
            header=0,
            sep=';'
        )
        self.data_dict = {'books': self.books, 'users': self.users, 'ratings': self.ratings}

    def test_case_1(self):
        pipeline = Pipeline([
            ('sparse', SparseMatrixCreator()),
            ('fit', SVDRecommender(n_components=2))
        ])

        preprecessed_data_dict = preprocessing(self.data_dict, True, 0, 0)

        tds = TrainTestSplitter(preprecessed_data_dict, 10, 0.2)

        train, test = next(tds.__iter__())

        pipeline.fit(train)

        y_pred = pipeline._final_estimator.predict(test)

        y_true = test['ratings']['Book-Rating'].values

        print(rmse(y_true, y_pred))


class TestNMFPipeline(unittest.TestCase):

    def setUp(self):
        enc = 'Windows-1251'
        addr = '../../notebooks/data/BX-{}.csv'
        self.ratings = pd.read_csv(
            filepath_or_buffer=addr.format('Book-Ratings'),
            error_bad_lines=False,
            warn_bad_lines=False,
            low_memory=False,
            encoding=enc,
            header=0,
            sep=';'
        )
        self.books = pd.read_csv(
            filepath_or_buffer=addr.format('Books'),
            error_bad_lines=False,
            warn_bad_lines=False,
            low_memory=False,
            encoding=enc,
            header=0,
            sep=';'
        )
        self.users = pd.read_csv(
            filepath_or_buffer=addr.format('Users'),
            error_bad_lines=False,
            warn_bad_lines=False,
            low_memory=False,
            encoding=enc,
            header=0,
            sep=';'
        )
        self.data_dict = {'books': self.books, 'users': self.users, 'ratings': self.ratings}

    def test_case_1(self):
        pipeline = Pipeline([
            ('sparse', SparseMatrixCreator()),
            ('fit', NMFRecommender(n_components=2))
        ])

        preprecessed_data_dict = preprocessing(self.data_dict, True, 0, 0)

        tds = TrainTestSplitter(preprecessed_data_dict, 10, 0.2)

        train, test = next(tds.__iter__())

        pipeline.fit(train)

        y_pred = pipeline._final_estimator.predict(test)

        y_true = test['ratings']['Book-Rating'].values

        print(rmse(y_true, y_pred))


if __name__ == '__main__':
    unittest.main()
