

def preprocessing(data_dict, is_explicit, user_ratings_count_threshold, book_ratings_count_threshold):
    books = data_dict['books']
    users = data_dict['users']
    ratings = data_dict['ratings']

    preprocessed_data_dict = {}

    # first thing we need to do is to decide whether we have to get
    # rid of zero ratings (it depends on is_explicit flag):
    if is_explicit:
        ratings = ratings[ratings['Book-Rating'] > 0]
    else:
        ratings['Book-Rating'] = 1
    # =========================================================================
    # get rid of the trash in some features:
    books = books[~books['Year-Of-Publication'].isin(['DK Publishing Inc', 'Gallimard'])]
    # here we have to perform some data filtration such as:
    # =========================================================================
    # - delete rating of the books we know nothing about:
    ratings = ratings[ratings.ISBN.isin(books.ISBN)]
    # =========================================================================
    # - delete rating of the users we know nothing about:
    ratings = ratings[ratings['User-ID'].isin(users['User-ID'])]
    # =========================================================================
    # also it would be good to get rid of useless features (Image-URL-S, Image-URL-M, Image-URL-L):
    books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
    # =========================================================================
    # now we are going to delete those users which have
    # rated/clicked on the amount of books below the threshold specified:
    ratings_grouped_by_user = ratings.groupby('User-ID').agg(len)
    users_needed = ratings_grouped_by_user[ratings_grouped_by_user.ISBN > user_ratings_count_threshold].index.values
    ratings = ratings[ratings['User-ID'].isin(users_needed)]
    # =========================================================================
    # the same filtration we will apply to items:
    ratings_grouped_by_book = ratings.groupby('ISBN').agg(len)
    books_needed = ratings_grouped_by_book[ratings_grouped_by_book['User-ID'] > book_ratings_count_threshold].index.values
    ratings = ratings[ratings.ISBN.isin(books_needed)]

    preprocessed_data_dict['books'] = books
    preprocessed_data_dict['users'] = users
    preprocessed_data_dict['ratings'] = ratings

    return preprocessed_data_dict
