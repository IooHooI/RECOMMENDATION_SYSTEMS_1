from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from scipy.sparse import coo_matrix


class SparseMatrixCreator(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        books = data['books']
        users = data['users']
        ratings = data['ratings']
        # Create a dictionary for users with pairs of type userID -> row number (in sparse matrix)
        user_ids_dict = dict(enumerate(users['User-ID']))
        user_ids_dict = dict(zip(user_ids_dict.values(), user_ids_dict.keys()))
        # Create a dictionary for items (books) with pairs of type itemID (ISBN) -> column number (in sparse matrix)
        book_isbns_dict = dict(enumerate(books['ISBN']))
        book_isbns_dict = dict(zip(book_isbns_dict.values(), book_isbns_dict.keys()))
        # Map users on their row numbers in sparse matrix
        columns = ratings['ISBN'].map(book_isbns_dict).values
        # Map items on their column numbers in sparse matrix
        rows = ratings['User-ID'].map(user_ids_dict).values
        # Create a separate array with ratings
        data = ratings['Book-Rating'].values
        # Create a sparse matrix
        sparse_matrix = coo_matrix((data, (rows, columns)), shape=(len(users), len(books)))
        return {'sparse_matrix': sparse_matrix, 'user_ids_dict': user_ids_dict, 'book_isbns_dict': book_isbns_dict}
