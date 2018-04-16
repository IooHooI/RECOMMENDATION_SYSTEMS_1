import numpy as np


class TrainTestSplitter:
    """
    Сплиттер дробит данные следующим образом:
    для каждого пользователя, который оценил количество книг больше,
    чем user_ratings_count_threshold, мы берем долю оценок, равную percent.
    Из этих данных формируется тестовая выборка.
    """

    def __init__(self, data_dict, user_ratings_count_threshold, percent):
        self.books = data_dict['books']
        self.users = data_dict['users']
        self.ratings = data_dict['ratings']

        self.user_ratings_count_threshold = user_ratings_count_threshold
        self.percent = percent

    def __iter__(self):
        ratings_grouped_by_user = self.ratings.groupby('User-ID').agg(len)

        users_needed = ratings_grouped_by_user[
            ratings_grouped_by_user.ISBN > self.user_ratings_count_threshold
            ].index.values

        test_ratings = []

        for user in users_needed:
            user_ratings = np.array(self.ratings[self.ratings['User-ID'] == user].index.values)
            test_ratings += user_ratings[
                np.random.randint(
                    len(user_ratings),
                    size=int(len(user_ratings) * self.percent)
                )
            ].tolist()
        test = {
            'books': self.books, 'users': self.users, 'ratings': self.ratings[self.ratings.index.isin(test_ratings)]
        }
        train = {
            'books': self.books, 'users': self.users, 'ratings': self.ratings[~self.ratings.index.isin(test_ratings)]
        }
        yield train, test
