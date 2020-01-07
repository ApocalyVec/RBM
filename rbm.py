# Boltzmann Machine

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

all_movie = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
all_users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
all_ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# prepare the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
# prepare the test set and the test set
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')


# Getting the number of users and movies
def get_num_user_movies(training_set, test_set):
    nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
    nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))
    return nb_users, nb_movies


# converting the data into an array with users in lines and movies in columns
def convert_bm_matrix(data, nb_users, nb_movies):
    """
    return a matrix of user's movie ratings. A row is a user's rating to all the movies. A column is a movie's rating
    from all the users.

    If using the function multiple times, you need to make sure that the user_id and movie_id across all
    of your dataset are consistent.
    :param data: the dataset that is going to be convert to user-movie_rating matrix
    :param nb_users: the number of users in the dataset
    :param nb_movies: the number of movies in the dataset
    :return: a matrix of user's movie ratings
    """
    new_data = []

    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings  # get all the rated movies indexes
        new_data.append(ratings)

    return new_data


num_users, num_movies = get_num_user_movies(training_set, test_set)
training_set = convert_bm_matrix(training_set, nb_users=num_users, nb_movies=num_movies)