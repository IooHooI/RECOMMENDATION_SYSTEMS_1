import unittest

import pandas as pd

from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import Dataset
from surprise import Reader

from source.code.utils import preprocessing
from source.code.evaluate import my_cross_validation


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
        algo = SVD()
        reader = Reader(rating_scale=(1, 10))
        preprocessed_data_dict = preprocessing(self.data_dict, True, 50, 50)
        preprocessed_data_dict['ratings'] = preprocessed_data_dict['ratings'].rename(
            {
                'User-ID': 'userID',
                'ISBN': 'itemID',
                'Book-Rating': 'rating'
            },
            axis='columns'
        )
        data = Dataset.load_from_df(preprocessed_data_dict['ratings'][['userID', 'itemID', 'rating']], reader)
        my_cross_validation(algo, data, k=5, threshold=7, n_splits=5, verbose=True)

    def test_case_2(self):
        algo = SVDpp()
        reader = Reader(rating_scale=(1, 10))
        preprocessed_data_dict = preprocessing(self.data_dict, False, 100, 100)
        preprocessed_data_dict['ratings'] = preprocessed_data_dict['ratings'].rename(
            {
                'User-ID': 'userID',
                'ISBN': 'itemID',
                'Book-Rating': 'rating'
            },
            axis='columns'
        )
        data = Dataset.load_from_df(preprocessed_data_dict['ratings'][['userID', 'itemID', 'rating']], reader)
        my_cross_validation(algo, data, k=5, threshold=7, n_splits=5, verbose=True)


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
        algo = NMF()
        reader = Reader(rating_scale=(1, 10))
        preprocessed_data_dict = preprocessing(self.data_dict, True, 50, 50)
        preprocessed_data_dict['ratings'] = preprocessed_data_dict['ratings'].rename(
            {
                'User-ID': 'userID',
                'ISBN': 'itemID',
                'Book-Rating': 'rating'
            },
            axis='columns'
        )
        data = Dataset.load_from_df(preprocessed_data_dict['ratings'][['userID', 'itemID', 'rating']], reader)
        my_cross_validation(algo, data, k=5, threshold=7, n_splits=5, verbose=True)


if __name__ == '__main__':
    unittest.main()
