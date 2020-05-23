# Boltzmann Machine

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


class RBM:
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)  # bias for the hidden nodes, one addition dimension for batch
        self.b = torch.randn(1, nv)  # bias for the visible nodes

    def sample_h(self, x):  # sample the activation of the hidden nodes given the value of the visible nodes
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)  # expand the bias according to batch, add the bias,
        p_h_given_v = torch.sigmoid(activation)  # probability of the hidden nodes are activated given the visible nodes
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):  # sample the value of the visible nodes given values of the hidden nodes
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):  # using contrasted divergence
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)  # keeping b to be a 2D tensor
        self.a += torch.sum((ph0 - phk), 0)


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


def convert_to_binary(data: torch.FloatTensor, threshold=3):
    rtn = data.clone()
    rtn[data == 0.] = -1.
    rtn[[x & y for (x, y) in zip([data > 0.], [data < threshold])]] = 0.
    rtn[data >= threshold] = 1.
    return rtn


# Script body ##########################################################################################
all_movie = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
all_users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
all_ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# prepare the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
# prepare the test set and the test set
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

num_users, num_movies = get_num_user_movies(training_set, test_set)
training_set = convert_bm_matrix(training_set, nb_users=num_users, nb_movies=num_movies)
test_set = convert_bm_matrix(test_set, nb_users=num_users, nb_movies=num_movies)

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

training_set = convert_to_binary(training_set)
test_set = convert_to_binary(test_set)

nv = num_movies
nh = 100  # tunable
batch_size = 100  # tunable
steps = 10
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for e in range(1, nb_epoch+1):
    train_loss = 0.
    s = 0.
    for id_user in range(0, num_users - batch_size, batch_size):
        v0 = training_set[id_user:id_user + batch_size]
        vk = v0
        ph0, _ = rbm.sample_h(v0)  # only take the activation probabilities
        for k in range(steps):  # k steps for the constrasted divergence
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]  # freeze the training on the none-ratings
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[0 <= v0] - vk[v0 >= 0]))
        s += 1.
    print('Epoch ' + str(e) + ': loss=' + str(train_loss / s))

# test the RMB
test_loss = 0.
s = 0.
for id_user in range(num_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]  # the target visible values
    if (len(vt[vt >= 0]) > 0):  # if the sample contains at least one -1 rating
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[0 <= vt] - v[vt >= 0]))
        s += 1.
print('Test loss=' + str(test_loss / s))