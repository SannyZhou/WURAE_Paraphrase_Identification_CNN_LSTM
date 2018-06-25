# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: function.py

import numpy as np
import tensorflow as tf

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def tanh_norm(x):
    n = np.linalg.norm(x)
    y = x - x**3
    return np.diag((1 - x**2)[:, 0 ] )/ n - np.dot(y, x.T) / n**3


def get_I_tensor(n):
    # m = []
    # while len(m) < n:
    #     m.append(np.ones(len(m) + 1))
    # return m[n - 1]
    return tf.ones([1, n])[0]


def sum_colum_tensor(m):
    row_n = tf.shape(m)[0]
    I = get_I_tensor(row_n)
    return tf.matmul(I, m)


def get_I(n):
    return np.ones([1, n])[0]


def sum_colum(m):
    row_n = m.shape[0]
    I = get_I(row_n)
    return np.dot(I, m)


def get_w_b(theta, embedding_size):
    # build and initialize by theta and embedding size
    # assert(theta.size == cls.paramater_size(embedding_size))
    offset = 0
    size = embedding_size * embedding_size
    We1 = theta[offset : offset + size].reshape(embedding_size, embedding_size)
    offset += size
    We2 = theta[offset: offset + size].reshape(embedding_size, embedding_size)
    offset += size
    be = theta[offset: offset + embedding_size].reshape(embedding_size, 1)
    offset += embedding_size

    Wd1 = theta[offset: offset + size].reshape(embedding_size, embedding_size)
    offset += size
    Wd2 = theta[offset: offset + size].reshape(embedding_size)
    offset += size
    bd1 = theta[offset: offset + embedding_size].reshape(embedding_size, 1)
    offset += embedding_size
    bd2 = theta[offset:].reshape(embedding_size, 1)
    return We1, We2, be, Wd1,Wd2, bd1, bd2