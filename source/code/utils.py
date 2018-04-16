

def preprocessing(data_dict, user_ratings_count_threshold, book_ratings_count_threshold):
    books = data_dict['books']
    users = data_dict['users']
    ratings = data_dict['ratings']

    preprocessed_data_dict = {}
    # здесь необходимо отбросить те записи, которые содержат 0 в поле Book-Rating,
    # т.е. отбросить отсутствие оценки книги I пользователем U:
    ratings = ratings[ratings['Book-Rating'] > 0]
    # далее необходимо провести все предварительные фильтрации с данными, такие как:
    # - удаление оценок для тех книг, о которых ничего неизвестно;
    ratings = ratings[ratings.ISBN.isin(books.ISBN)]
    # - удаление оценок для тех пользователей, о которых ничего неизвестно;
    ratings = ratings[ratings['User-ID'].isin(users['User-ID'])]
    # - удаление пользователей, которые никак ничего не оценили;
    users = users[users['User-ID'].isin(ratings['User-ID'])]
    # - удаление книг, которые никто никак не оценил;
    books = books[books.ISBN.isin(ratings.ISBN)]
    # =========================================================================
    # теперь еще хорошо бы избавиться от мусорных полей со ссылками на рисунки обложек:
    books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
    # =========================================================================
    # теперь надо избавиться от тех пользователей, которые оценили количество книг меньше указанного порога
    # и от тех книг, которые были оценены количеством пользователей меньше указанного порога:

    ratings_grouped_by_user = ratings.groupby('User-ID').agg(len)
    ratings_grouped_by_book = ratings.groupby('ISBN').agg(len)

    users_needed = ratings_grouped_by_user[ratings_grouped_by_user.ISBN > user_ratings_count_threshold].index.values
    books_needed = ratings_grouped_by_book[ratings_grouped_by_book['User-ID'] > book_ratings_count_threshold].index.values

    ratings = ratings[ratings['User-ID'].isin(users_needed)]
    ratings = ratings[ratings.ISBN.isin(books_needed)]

    users = users[users['User-ID'].isin(users_needed)]
    books = books[books.ISBN.isin(books_needed)]

    preprocessed_data_dict['books'] = books
    preprocessed_data_dict['users'] = users
    preprocessed_data_dict['ratings'] = ratings

    return preprocessed_data_dict