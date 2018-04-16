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
        # Создаем словарь для пользователей с парами типа (ID пользователя -> номер строки в разреженной матрице)
        user_ids_dict = dict(enumerate(users['User-ID']))
        user_ids_dict = dict(zip(user_ids_dict.values(), user_ids_dict.keys()))
        # Создаем словарь для книг с парами типа (ISBN книги -> номер столбца в разреженной матрице)
        book_isbns_dict = dict(enumerate(books['ISBN']))
        book_isbns_dict = dict(zip(book_isbns_dict.values(), book_isbns_dict.keys()))
        # Отображаем пользователей на их номера в матрице
        columns = ratings['ISBN'].map(book_isbns_dict).values
        # Отображаем книги на их номера в матрице
        rows = ratings['User-ID'].map(user_ids_dict).values
        # Выделяем оценки в отдельный массив
        data = ratings['Book-Rating'].values
        # Создаемм разреженную матрицу
        sparse_matrix = coo_matrix((data, (rows, columns)), shape=(len(users), len(books)))
        return {'sparse_matrix': sparse_matrix, 'user_ids_dict': user_ids_dict, 'book_isbns_dict': book_isbns_dict}