from sklearn.decomposition.nmf import NMF
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from tqdm import tqdm

import numpy as np


class NMFRecommender(BaseEstimator, RegressorMixin):

    def __init__(self, n_components):
        self.nmf = NMF(n_components=2, init='random', random_state=0)
        self.user_ids_dict = {}
        self.book_isbns_dict = {}

    def fit(self, X, y=None):
        self.sparse_matrix = X['sparse_matrix']
        self.user_ids_dict = X['user_ids_dict']
        self.book_isbns_dict = X['book_isbns_dict']
        self.nmf.fit(X['sparse_matrix'])

    def predict(self, X, y=None):
        ratings = X['ratings']
        user_representations = self.nmf.transform(self.sparse_matrix)
        book_representations = self.nmf.components_
        estimations = []
        for i in tqdm(range(len(ratings))):
            estimation = np.dot(
                user_representations[self.user_ids_dict[ratings.iloc[i]['User-ID']]],
                book_representations
            )[self.book_isbns_dict[ratings.iloc[i]['ISBN']]]
            estimations.append(estimation)
        return estimations

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X, y)
