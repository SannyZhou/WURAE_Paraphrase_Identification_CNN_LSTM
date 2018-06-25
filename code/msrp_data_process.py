# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: msrp_data_process.py

from gensim.models import word2vec
import os
import pandas as pd
import numpy as np
import math
from nltk.tokenize import RegexpTokenizer
from keras.engine import Layer
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras import optimizers
from keras.regularizers import l2
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.engine import Layer, InputSpec
import tensorflow as tf
from sklearn import metrics
import keras.backend.tensorflow_backend as KTF
from keras import callbacks
from keras.layers import Bidirectional, multiply, add, subtract
import pickle
import re
from new_features import *


# Build word vector
def get_word_embedding():
    print("Build word embedding.")
    try:
        word_vectors = word2vec.Word2VecKeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
        print("Finish building word embedding.")
    except Exception as e:
        print("Fail in building word embedding", e)
    return word_vectors


# change sentence to matrix
def sentence_to_matrix(s_list):
    print("Build sentence matrix.")
    sentence_matrix = []
    for sentence in s_list:
        m = []
        for word in sentence:
            if word in word_embedding:
                m.append(np.array(word_embedding[word]))
            else:
                m.append((np.random.rand(300) / 5 - 0.1))
            # also can test oov word embedding in range of(-0.1, 0.1)
        while len(m) < 35:
            m.append(np.array([0] * 300))
        sentence_matrix.append(np.array(m))
    print("Finish building sentence matrix.")
    return np.array(sentence_matrix)


# reshape node feature matrix
def node_matrix(node_feature_list):
    print('Reshape node feature matrix.')
    node_feature_matrix = []
    for node_feature in node_feature_list:
        m = []
        for node in node_feature:
            m.append(np.array(node))
        while len(m) < 39:
            m.append(np.array([0] * 300))
        node_feature_matrix.append(np.array(m))
    print('Finish reshaping the node feature matrix.')
    return np.array(node_feature_matrix)


# calculate the similarity matrix of words in two sentences
def words_similarity_martix(s1_matrix, s2_matrix):
    # calculate the cos distance of words in each pair of sentence
    # the output matrix would be constructed by (35,35)
    print('Build word similarity matrix.')
    word_matrix = []
    sentence_num = s1_matrix.shape[0]
    print(s1_matrix.shape)
    for i in range(sentence_num):
        sentence_pair_matrix = []
        for word_vector_1 in s1_matrix[i]:
            word_to_words_vector = []
            for word_vector_2 in s2_matrix[i]:
                n = float(np.dot(word_vector_1.T, word_vector_2))
                denom = np.linalg.norm(word_vector_1) * np.linalg.norm(word_vector_2)
                if denom == 0.0:
                    sim = 0
                else:
                    cos = n / denom
                    sim = 0.5 + 0.5 * cos
                word_to_words_vector.append(sim)
            sentence_pair_matrix.append(np.array(word_to_words_vector))
        word_matrix.append(np.array(sentence_pair_matrix))
    print('Finish building word similarity matrix.')
    return np.array(word_matrix)


# calculate the similarity matrix of node in two sentences
def nodes_similarity_matrix(s1_node, s2_node):
    # calculate the cosin distance of nodes in each pair of sentences
    # the output matrix would be constructed by (35,35)
    print('Build nodes similarity matrix.')
    node_matrix = []
    sentence_num = s1_node.shape[0]
    print(s1_node.shape)
    for i in range(sentence_num):
        sentence_pair_matrix = []
        for node_vec_1 in s1_node[i]:
            node_to_nodes_vector = []
            for node_vec_2 in s2_node[i]:
                n = float(np.dot(node_vec_1.T, node_vec_2))
                denom = np.linalg.norm(node_vec_1) * np.linalg.norm(node_vec_2)
                if denom == 0.0:
                    sim = 0
                else:
                    cos = n / denom
                    sim = 0.5 + 0.5 * cos
                node_to_nodes_vector.append(sim)
            sentence_pair_matrix.append(np.array(node_to_nodes_vector))
        node_matrix.append(np.array(sentence_pair_matrix))
    print('Finish building nodes similarity matrix.')
    return np.array(node_matrix)


# get the feature of number occurs in the two sentences
def get_num_feature(s1, s2):
    num_feature = [0, 0, 0]
    p = re.compile('[-+]?[0-9]*\.?[0-9]+')
    s1_n = p.findall(s1)
    s2_n = p.findall(s2)

    if s1_n and s2_n:
        num_feature[0] = 0
    # both don't have numbers
    elif not s1_n and not s2_n:
        num_feature[0] = 1
        return num_feature
    else:
        return num_feature
    s1_n_set = set(s1_n)
    s2_n_set = set(s2_n)
    # have same numbers
    if s1_n_set & s2_n_set == s1_n_set | s2_n_set:
        num_feature[0] = 1
        num_feature[1] = 1
    interset = s1_n_set & s2_n_set
    # numbers in one sentence is the strict subset of another one
    if s1_n_set == interset or s2_n_set == interset:
        num_feature[2] = 1
    else:
        num_feature[2] = 0
    return np.array(num_feature)


# load the data of microsoft research paraphrase corpus and double the train data
def load_msrp_train_data(filename, nodefeaturepickle):
    # sim_wdnt = pos_tagger_similarity('sentence_msr_paraphrase_trainparsed.txt')
    df = pd.DataFrame()
    label, sid, s1, s2, s1_node, s2_node, number_feature, s_mt_feature, s_tf_idf_feature, s_lcs_feature, s_minedit_feature\
        = [], [], [], [], [], [], [], [], [], [], []
    nf = open(nodefeaturepickle, 'rb')
    nodefeature_matrix = pickle.load(nf)
    nf.close()
    print(len(nodefeature_matrix))
    print(np.array(nodefeature_matrix).shape)
    f = open(filename, 'r')
    f.readline()
    tokenizer = RegexpTokenizer(r'\w+')
    index = 0
    for l in f.readlines():
        line = l.strip().split('\t')
        label.append(int(line[0]))
        sid.append([line[1], line[2]])
        number_feature.append(get_num_feature(line[3], line[4]))
        s1.append(tokenizer.tokenize(line[3].lower()))
        s2.append(tokenizer.tokenize(line[4].lower()))
        s_mt_feature.append(get_sentence_bleu(s1[index], s2[index]))
        s_tf_idf_feature.append([eng_tiidf_feature(line[3], line[4])])
        s_lcs_feature.append([lcs(s1[index], s2[index])])
        s_minedit_feature.append([minimumEditDistance(s1[index], s2[index])])
        s1_node.append(nodefeature_matrix[index][0])
        s2_node.append(nodefeature_matrix[index][1])
        index += 1
    f.close()

    df['label'], df['sid'], df['s1'], df['s2'], df['s1_node'], df['s2_node'],\
    df['num_feature'], df['mt_feature'], df['tf_idf_feature'], df['lcs'], df['minedit'],\
     = label, sid, s1, s2, s1_node, s2_node, number_feature, s_mt_feature, s_tf_idf_feature,\
       s_lcs_feature, s_minedit_feature
    df = df.sample(frac=1.0)
    df = df.reset_index(drop=True)
    df_train = df.iloc[0:int(0.9 * len(df))]
    df_val = df.iloc[int(0.9 * len(df) + 1):]
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    len_val = len(df_val)
    df_train = df
    len_train = len(df_train)
    for iter in range(len_train):
        sid = df_train.loc[iter]['sid']
        sid = [sid[1], sid[0]]
        label = df_train.loc[iter]['label']
        s1 = df_train.loc[iter]['s2']
        s2 = df_train.loc[iter]['s1']
        s1_node = df_train.loc[iter]['s2_node']
        s2_node = df_train.loc[iter]['s1_node']
        number_feature = df_train.loc[iter]['num_feature']
        s_mt_feature = get_sentence_bleu(s1, s2)
        s_minedit_feature = df_train.loc[iter]['minedit']
        s_lcs_feature = df_train.loc[iter]['lcs']
        s_tf_idf_feature = df_train.loc[iter]['tf_idf_feature']
        tmp_df = {'label': label, 'sid': sid, 's1': s1, 's2': s2, 's1_node': s1_node, 's2_node': s2_node,
                  'num_feature': number_feature, 'mt_feature': s_mt_feature, 'tf_idf_feature': s_tf_idf_feature,
                  'lcs': s_lcs_feature, 'minedit': s_minedit_feature}
        df_train = df_train.append(tmp_df, ignore_index=True)
    print(df_train.loc[0], df_train.loc[len_train])
    df_train = df_train.sample(frac=1.0)
    df_train = df_train.reset_index(drop=True)
    print(len(df_train))
    for iter in range(len_val):
        sid = df_val.loc[iter]['sid']
        sid = [sid[1], sid[0]]
        label = df_val.loc[iter]['label']
        s1 = df_val.loc[iter]['s2']
        s2 = df_val.loc[iter]['s1']
        s1_node = df_val.loc[iter]['s2_node']
        s2_node = df_val.loc[iter]['s1_node']
        number_feature = df_val.loc[iter]['num_feature']
        s_mt_feature = get_sentence_bleu(s1, s2)
        s_minedit_feature = df_val.loc[iter]['minedit']
        s_lcs_feature = df_val.loc[iter]['lcs']
        s_tf_idf_feature = df_val.loc[iter]['tf_idf_feature']
        tmp_df = {'label': label, 'sid': sid, 's1': s1, 's2': s2, 's1_node': s1_node, 's2_node': s2_node,
                  'num_feature': number_feature, 'mt_feature': s_mt_feature, 'tf_idf_feature': s_tf_idf_feature,
                  'lcs': s_lcs_feature, 'minedit': s_minedit_feature}
        df_val = df_val.append(tmp_df, ignore_index=True)
    print(df_val.loc[0], df_val.loc[len_val])
    df_val = df_val.sample(frac=1.0)
    df_val = df_val.reset_index(drop=True)
    print(len(df_val))
    return (df_train['label'], df_train['sid'],
            df_train['s1'], df_train['s2'],
            df_train['s1_node'], df_train['s2_node'],
            df_train['num_feature'], df_train['mt_feature'], df_train['tf_idf_feature'],
            df_train['lcs'], df_train['minedit'],
            df_val['label'], df_val['sid'],
            df_val['s1'], df_val['s2'],
            df_val['s1_node'], df_val['s2_node'],
            df_val['num_feature'], df_val['mt_feature'], df_val['tf_idf_feature'],
            df_val['lcs'], df_val['minedit']
            )


# load the data of microsoft research paraphrase corpus and double the test data
def load_msrp_test_data(filename, nodefeaturepickle):
    # sim_wdnt = pos_tagger_similarity('sentence_msr_paraphrase_testparsed.txt')
    df = pd.DataFrame()
    label, sid, s1, s2, s1_node, s2_node, number_feature, s_mt_feature, s_tf_idf_feature, s_lcs_feature, s_minedit_feature\
        = [], [], [], [], [], [], [], [], [], [], []
    nf = open(nodefeaturepickle, 'rb')
    nodefeature_matrix = pickle.load(nf)
    nf.close()
    f = open(filename, 'r')
    f.readline()
    tokenizer = RegexpTokenizer(r'\w+')
    index = 0
    for l in f.readlines():
        line = l.strip().split('\t')
        label.append(int(line[0]))
        sid.append([line[1], line[2]])
        s1.append(tokenizer.tokenize(line[3].lower()))
        s2.append(tokenizer.tokenize(line[4].lower()))
        s1_node.append(nodefeature_matrix[index][0])
        s2_node.append(nodefeature_matrix[index][1])
        number_feature.append(get_num_feature(line[3], line[4]))
        s_mt_feature.append(get_sentence_bleu(s1[index], s2[index]))
        s_tf_idf_feature.append([eng_tiidf_feature(line[3], line[4])])
        s_lcs_feature.append([lcs(s1[index], s2[index])])
        s_minedit_feature.append([minimumEditDistance(s1[index], s2[index])])
        index += 1
    f.close()
    df['label'], df['sid'], df['s1'], df['s2'], df['s1_node'], df['s2_node'],\
    df['num_feature'], df['mt_feature'],df['tf_idf_feature'], df['lcs'], df['minedit']\
    = label, sid, s1, s2, s1_node, s2_node, number_feature, s_mt_feature,\
      s_tf_idf_feature, s_lcs_feature, s_minedit_feature
    return df['label'], df['sid'], df['s1'], df['s2'], df['s1_node'], df['s2_node'],\
           df['num_feature'], df['mt_feature'], df['tf_idf_feature'],  df['lcs'], df['minedit']


def to_narray(num_f):
    res = []
    for nf in num_f:
        res.append(np.array(nf))
    return np.array(res)


def cos_sum_embedding(s1_list, s2_list):
    s1_tmp_embedding = sum(s1_list)
    s2_tmp_embedding = sum(s2_list)
    n = float(np.dot(s1_tmp_embedding.T, s2_tmp_embedding))
    denom = np.linalg.norm(s1_tmp_embedding) * np.linalg.norm(s2_tmp_embedding)
    if denom == 0.0:
        sim = 0
    else:
        cos = n / denom
        sim = 0.5 + 0.5 * cos
    return sim


def list_cos_sum(s1_matrix, s2_matrix):
    length_data = len(s1_matrix)
    res = []
    for s_index in range(length_data):
        res.append([cos_sum_embedding(s1_matrix[s_index], s2_matrix[s_index])])
    return res


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
    # KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 3, 'cpu': 5}, gpu_options=gpu_options)))
    os.system('echo $CUDA_VISIBLE_DEVICES')
    train_label, train_sid, train_s1, train_s2, train_s1_node, train_s2_node,\
    train_num_feature, train_mt_feature, train_tf_idf_feature, train_lcs, train_minedit,\
    val_label, val_sid, val_s1, val_s2, val_s1_node, val_s2_node,\
    val_num_feature, val_mt_feature, val_tf_idf_feature, val_lcs, val_minedit,\
        = load_msrp_train_data(
    '../data/msr_paraphrase_train.txt', 'torchweights/english_wurae_sentence_msr_paraphrase_trainparsed_nodeFeature.pickle')
    test_label, test_sid, test_s1, test_s2, test_s1_node, test_s2_node,\
    test_num_feature, test_mt_feature, test_tf_idf_feature, test_lcs, test_minedit, \
        = load_msrp_test_data(
    '../data/msr_paraphrase_test.txt', 'torchweights/english_wurae_sentence_msr_paraphrase_testparsed_nodeFeature.pickle')

    train_num_feature = to_narray(train_num_feature)
    test_num_feature = to_narray(test_num_feature)
    val_num_feature = to_narray(val_num_feature)
    train_mt_feature = to_narray(train_mt_feature)
    val_mt_feature = to_narray(val_mt_feature)
    test_mt_feature = to_narray(test_mt_feature)
    train_tf_idf_feature = to_narray(train_tf_idf_feature)
    val_tf_idf_feature = to_narray(val_tf_idf_feature)
    test_tf_idf_feature = to_narray(test_tf_idf_feature)
    train_lcs = to_narray(train_lcs)
    test_lcs = to_narray(test_lcs)
    val_lcs = to_narray(val_lcs)
    train_minedit = to_narray(train_minedit)
    test_minedit = to_narray(test_minedit)
    val_minedit = to_narray(val_minedit)
    print(train_num_feature.shape, train_mt_feature.shape, train_tf_idf_feature.shape, train_lcs.shape, train_minedit.shape)
    # os.system('nvidia-smi')
    # os.system('ps')
    word_embedding = get_word_embedding()
    train_s1_matrix = sentence_to_matrix(train_s1)
    train_s2_matrix = sentence_to_matrix(train_s2)
    print(train_s1_matrix.shape)
    train_word_matrix = words_similarity_martix(train_s1_matrix, train_s2_matrix)
    train_word_matrix_t = train_word_matrix.transpose(0, 2, 1)
    train_s1_node = node_matrix(train_s1_node)
    train_s2_node = node_matrix(train_s2_node)
    print(train_word_matrix.shape)
    print(train_word_matrix[0].shape)

    train_node_matrix = nodes_similarity_matrix(train_s1_node, train_s2_node)
    print(train_node_matrix.shape)
    train_node_matrix_t = train_node_matrix.transpose(0, 2, 1)
    val_s1_matrix = sentence_to_matrix(val_s1)
    val_s2_matrix = sentence_to_matrix(val_s2)
    val_word_matrix = words_similarity_martix(val_s1_matrix, val_s2_matrix)
    val_word_matrix_t = val_word_matrix.transpose(0, 2, 1)
    val_s1_node = node_matrix(val_s1_node)
    val_s2_node = node_matrix(val_s2_node)
    val_node_matrix = nodes_similarity_matrix(val_s1_node, val_s2_node)
    val_node_matrix_t = val_node_matrix.transpose(0, 2, 1)
    test_s1_matrix = sentence_to_matrix(test_s1)
    test_s2_matrix = sentence_to_matrix(test_s2)
    test_word_matrix = words_similarity_martix(test_s1_matrix, test_s2_matrix)
    test_word_matrix_t = test_word_matrix.transpose(0, 2, 1)
    test_s1_node = node_matrix(test_s1_node)
    test_s2_node = node_matrix(test_s2_node)
    test_node_matrix = nodes_similarity_matrix(test_s1_node, test_s2_node)
    test_node_matrix_t = test_node_matrix.transpose(0, 2, 1)

    train_set = {'s1':train_s1, 's2':train_s2, 'label':train_label, 's1_matrix':train_s1_matrix, 's2_matrix':train_s2_matrix,
             'word_matrix':train_word_matrix, 'word_matrix_t':train_word_matrix_t,
             'node_matrix': train_node_matrix, 'node_matrix_t': train_node_matrix_t,
             'nf': train_num_feature, 'mt': train_mt_feature, 'tfidf': train_tf_idf_feature, 'lcs': train_lcs,
             'minedit': train_minedit}

    test_set = {'s1': test_s1, 's2': test_s2, 'label': test_label, 's1_matrix': test_s1_matrix,
                's2_matrix': test_s2_matrix,
                'word_matrix': test_word_matrix, 'word_matrix_t': test_word_matrix_t,
                'node_matrix': test_node_matrix, 'node_matrix_t': test_node_matrix_t,
                'nf': test_num_feature, 'mt': test_mt_feature, 'tfidf': test_tf_idf_feature, 'lcs': test_lcs,
                'minedit': test_minedit}

    val_set = {'s1': val_s1, 's2': val_s2, 'label': val_label, 's1_matrix': val_s1_matrix, 's2_matrix': val_s2_matrix,
               'word_matrix': val_word_matrix, 'word_matrix_t': val_word_matrix_t,
               'node_matrix': val_node_matrix, 'node_matrix_t': val_node_matrix_t,
               'nf': val_num_feature, 'mt': val_mt_feature, 'tfidf': val_tf_idf_feature, 'lcs': val_lcs,
               'minedit': val_minedit}

    train_f = open('msrpc_train_set.pkl', 'wb')
    pickle.dump(train_set, train_f)
    train_f.close()
    test_f = open('msrpc_test_set.pkl', 'wb')
    pickle.dump(test_set, test_f)
    test_f.close()
    val_f = open('msrpc_val_set.pkl', 'wb')
    pickle.dump(val_set, val_f)
    val_f.close()
