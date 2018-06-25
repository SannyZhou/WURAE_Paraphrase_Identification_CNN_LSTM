# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: chinese_article_level_parahraseDetection.py

import os
import pandas as pd
import numpy as np
import math
from keras.engine import Layer
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate, Bidirectional, subtract, multiply
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
import pickle
import re
from keras.utils import plot_model
import argparse


class KMaxPooling(Layer):
    #K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    #TensorFlow backend.
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)


# LearningRateScheduler function
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def learn(train_s1_matrix, train_s2_matrix, test_s1_matrix, test_s2_matrix, val_s1_matrix,
          val_s2_matrix, train_label, test_label, val_label, train_word_matrix,
          train_word_matrix_transposed, test_word_matrix, test_word_matrix_transposed,
          val_word_matrix, val_word_matrix_transposed, train_num_feature, test_num_feature, val_num_feature):
    # convolution layer to get correlation between sentences(like n-gram)
    # convolute every sentence separately
    input_article_1 = Input(shape=(55, 300), name='input_article_1')
    input_article_2 = Input(shape=(55, 300), name='input_article_2')

    # shared convolution layer for 2-gram, 3-gram, 5-gram, 7-gram
    shared_convolution_1 = Convolution1D(filters=300, kernel_size=2, strides=1, padding='same', input_shape=(55, 300))
    shared_convolution_2 = Convolution1D(filters=300, kernel_size=3, strides=1, padding='same', input_shape=(55, 300))
    shared_convolution_3 = Convolution1D(filters=300, kernel_size=5, strides=1, padding='same', input_shape=(55, 300))
    shared_convolution_4 = Convolution1D(filters=300, kernel_size=7, strides=1, padding='same', input_shape=(55, 300))

    # reuse the same layer with the two inputs
    convolution_output_1_1 = shared_convolution_1(input_article_1)
    convolution_output_1_2 = shared_convolution_1(input_article_2)
    convolution_output_2_1 = shared_convolution_2(input_article_1)
    convolution_output_2_2 = shared_convolution_2(input_article_2)
    convolution_output_3_1 = shared_convolution_3(input_article_1)
    convolution_output_3_2 = shared_convolution_3(input_article_2)
    convolution_output_4_1 = shared_convolution_4(input_article_1)
    convolution_output_4_2 = shared_convolution_4(input_article_2)

    # shared max pooling layer to get the semantic feature and reduce the cost of calculation
    shared_max_pooling_1 = MaxPooling1D(pool_size=2, strides=1)
    shared_max_pooling_2 = MaxPooling1D(pool_size=2, strides=1)
    shared_max_pooling_3 = MaxPooling1D(pool_size=2, strides=1)
    shared_max_pooling_4 = MaxPooling1D(pool_size=2, strides=1)
    act_f_1 = Activation('relu')

    # reuse the same layer with the two sentence
    max_pooling_output_0_1 = shared_max_pooling_1(input_article_1)
    max_pooling_output_0_2 = shared_max_pooling_1(input_article_2)
    max_pooling_output_1_1 = shared_max_pooling_1(convolution_output_1_1)
    max_pooling_output_1_2 = shared_max_pooling_1(convolution_output_1_2)
    max_pooling_output_2_1 = shared_max_pooling_2(convolution_output_2_1)
    max_pooling_output_2_2 = shared_max_pooling_2(convolution_output_2_2)
    max_pooling_output_3_1 = shared_max_pooling_3(convolution_output_3_1)
    max_pooling_output_3_2 = shared_max_pooling_3(convolution_output_3_2)
    max_pooling_output_4_1 = shared_max_pooling_4(convolution_output_4_1)
    max_pooling_output_4_2 = shared_max_pooling_4(convolution_output_4_2)

    max_pooling_output_0_1 = act_f_1(max_pooling_output_0_1)
    max_pooling_output_0_2 = act_f_1(max_pooling_output_0_2)
    max_pooling_output_1_1 = act_f_1(max_pooling_output_1_1)
    max_pooling_output_1_2 = act_f_1(max_pooling_output_1_2)
    max_pooling_output_2_1 = act_f_1(max_pooling_output_2_1)
    max_pooling_output_2_2 = act_f_1(max_pooling_output_2_2)
    max_pooling_output_3_1 = act_f_1(max_pooling_output_3_1)
    max_pooling_output_3_2 = act_f_1(max_pooling_output_3_2)
    max_pooling_output_4_1 = act_f_1(max_pooling_output_4_1)
    max_pooling_output_4_2 = act_f_1(max_pooling_output_4_2)

    # concatencate the features from the four convolution-max-pooling layers
    convolution_feature_1 = concatenate([max_pooling_output_0_1, max_pooling_output_1_1,
                                         max_pooling_output_2_1, max_pooling_output_3_1, max_pooling_output_4_1],
                                        axis=-1)
    convolution_feature_2 = concatenate([max_pooling_output_0_2, max_pooling_output_1_2,
                                         max_pooling_output_2_2, max_pooling_output_3_2, max_pooling_output_4_2],
                                        axis=-1)

    # shared LSTM for semantic feature of each sentence
    shared_lstm_1 = Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(54, 1500)))
    shared_lstm_2 = LSTM(units=512, return_sequences=True)
    shared_lstm_3 = LSTM(units=1024, return_sequences=False)

    # reuse the same layer with the two features matrix from the two articles
    sequence_1 = shared_lstm_1(convolution_feature_1)
    sequence_2 = shared_lstm_1(convolution_feature_2)
    sequence_1 = shared_lstm_2(sequence_1)
    sequence_2 = shared_lstm_2(sequence_2)
    last_sequence_1 = shared_lstm_3(sequence_1)
    last_sequence_2 = shared_lstm_3(sequence_2)
    last_sequence_12_mul = multiply([last_sequence_1, last_sequence_2])
    last_sequence_12_sub = subtract([last_sequence_1, last_sequence_2])
    last_sequence_12_l2 = multiply([last_sequence_12_sub, last_sequence_12_sub])

    # concatencate the semantic feature from the lstm
    semantic_feature = concatenate(
        [last_sequence_1, last_sequence_2, last_sequence_12_mul, last_sequence_12_sub, last_sequence_12_l2], axis=-1)

    # sentences similarity matrix as new input for unit similarity feature
    input_articles_matrix = Input(shape=(55, 55), name='input_articles_matrix')
    input_articles_matrix_transposed = Input(shape=(55, 55), name='input_articles_matrix_transposed')

    # convolution for the unit similarity feature
    convolution_unit_1 = Convolution1D(filters=256, kernel_size=3, strides=1, input_shape=(55, 55))
    max_pooling_lexical_1 = MaxPooling1D(pool_size=2, strides=1)
    convolution_unit_2 = Convolution1D(filters=512, kernel_size=5, strides=1)
    max_pooling_lexical_2 = MaxPooling1D(pool_size=2, strides=1)
    convolution_unit_3 = Convolution1D(filters=1024, kernel_size=7, strides=1)

    # 15-max-pooling
    max_pooling_1_layer = KMaxPooling(15)
    act_f_2 = Activation('relu')

    # calculate the unit-level feature by the convolution
    convolution_unit_output_1_1 = convolution_unit_1(input_articles_matrix)
    max_pooling_lexical_output_1_1 = max_pooling_lexical_1(convolution_unit_output_1_1)
    max_pooling_lexical_output_1_1 = act_f_2(max_pooling_lexical_output_1_1)

    convolution_unit_output_1_2 = convolution_unit_2(max_pooling_lexical_output_1_1)
    max_pooling_lexical_1_2 = max_pooling_lexical_2(convolution_unit_output_1_2)
    max_pooling_lexical_1_2 = act_f_2(max_pooling_lexical_1_2)

    convolution_unit_output_1_3 = convolution_unit_3(max_pooling_lexical_1_2)
    max_pooling_1_output_1 = max_pooling_1_layer(convolution_unit_output_1_3)
    max_pooling_1_output_1 = act_f_2(max_pooling_1_output_1)

    convolution_unit_output_2_1 = convolution_unit_1(input_articles_matrix_transposed)
    max_pooling_lexical_output_2_1 = max_pooling_lexical_1(convolution_unit_output_2_1)
    max_pooling_lexical_output_2_1 = act_f_2(max_pooling_lexical_output_2_1)

    convolution_unit_output_2_2 = convolution_unit_2(max_pooling_lexical_output_2_1)
    max_pooling_lexical_2_2 = max_pooling_lexical_2(convolution_unit_output_2_2)
    max_pooling_lexical_2_2 = act_f_2(max_pooling_lexical_2_2)

    convolution_unit_output_2_3 = convolution_unit_3(max_pooling_lexical_2_2)
    max_pooling_1_output_2 = max_pooling_1_layer(convolution_unit_output_2_3)
    max_pooling_1_output_2 = act_f_2(max_pooling_1_output_2)

    # concatenate the unit similarity feature from the cnn
    unit_similarity_feature = concatenate([max_pooling_1_output_1, max_pooling_1_output_2], axis=-1)

    # number feature
    input_num_feature = Input(shape=(3, ), name='input_num_feature')

    # total feature
    feature = concatenate([semantic_feature, unit_similarity_feature], axis=-1)
    feature = concatenate([feature, input_num_feature], axis=-1)
    # classification
    f = Dense(1024, activation='relu')(feature)
    f = Dropout(0.1)(f)
    f = Dense(256, activation='relu')(f)
    f = Dropout(0.1)(f)
    f = Dense(32, activation='relu')(f)
    f = Dropout(0.1)(f)
    output_classification = Dense(1, activation='sigmoid', name='output_classification')(f)

    # define model
    model = Model(inputs=[input_article_1, input_article_2, input_articles_matrix,
                          input_articles_matrix_transposed, input_num_feature
                          ],
                  outputs=output_classification)
    lrate = callbacks.LearningRateScheduler(step_decay)
    optim = optimizers.Adadelta(lr=0.175, rho=0.95, epsilon=1e-7)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['acc'])
    # plot_model(model, to_file='chinese_PI_model.png')
    checkpoint_tmp = callbacks.ModelCheckpoint('checkpoint_chinese_paraphrase_detection')
    earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=patience_early_stopping)
    model.summary()
    reducelearnrate = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, epsilon=1e-07)
    model.summary()
    if if_test:
        model.load_weights('checkpoint_chinese_paraphrase_detection')
    else:
        model.fit([train_s1_matrix, train_s2_matrix, train_word_matrix, train_word_matrix_transposed,
                   train_num_feature], train_label,
                  epochs=512, batch_size=64, callbacks=[earlystop, reducelearnrate, checkpoint_tmp], verbose=2,
                  validation_data=([val_s1_matrix, val_s2_matrix, val_word_matrix, val_word_matrix_transposed,
                                    val_num_feature], val_label))

    pre = model.predict([test_s1_matrix, test_s2_matrix, test_word_matrix, test_word_matrix_transposed,
                         test_num_feature])
    score = model.evaluate([test_s1_matrix, test_s2_matrix, test_word_matrix, test_word_matrix_transposed,
                            test_num_feature], test_label, verbose=0)
    metric_name = model.metrics_names
    print('end of model')
    return pre, score, metric_name


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


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 3, 'cpu': 13}, gpu_options=gpu_options)))
    os.system('echo $CUDA_VISIBLE_DEVICES')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--early-stopping', type=int, default=15,
                            help='patience of early stopping')
    arg_parser.add_argument('-test', type=int, default=0,
                            help='1 for test, 0 for train')
    options = arg_parser.parse_args()
    patience_early_stopping = options.early_stopping
    if_test = options.test

    train_f = open('../data/sample_chinese_train_set_with_others.pkl', 'rb')
    # train_f = open('../data/chinese_train_set_with_others.pkl', 'rb')
    train_data = pickle.load(train_f)
    train_f.close()
    train_origin_sentence_embedding, train_repl_sentence_embedding, train_label, train_sen_similarity,\
    train_sen_similarity_t, train_nf =\
        train_data['origin_sentence_embedding'], train_data['repl_sentence_embedding'], train_data['label'],\
        train_data['similarity'],\
        train_data['word_matrix_t'], train_data['nf']

    val_f = open('../data/sample_chinese_train_set_with_others.pkl', 'rb')
    # val_f = open('../data/chinese_val_set_with_others.pkl', 'rb')
    val_data = pickle.load(val_f)
    val_f.close()
    val_origin_sentence_embedding, val_repl_sentence_embedding, val_label, val_sen_similarity, \
    val_sen_similarity_t, val_nf = \
        val_data['origin_sentence_embedding'], val_data['repl_sentence_embedding'], val_data['label'], \
        val_data['similarity'], \
        val_data['word_matrix_t'], val_data['nf']

    test_f  = open('../data/sample_chinese_train_set_with_others.pkl', 'rb')
    # test_f = open('../data/chinese_test_set_with_others.pkl', 'rb')
    test_data = pickle.load(test_f)
    test_f.close()
    test_origin_sentence_embedding, test_repl_sentence_embedding, test_label, test_sen_similarity, \
    test_sen_similarity_t, test_nf = \
        test_data['origin_sentence_embedding'], test_data['repl_sentence_embedding'], test_data['label'], \
        test_data['similarity'], \
        test_data['word_matrix_t'], test_data['nf']

    predicted_label, score, metric_name = learn(train_origin_sentence_embedding,
                                                train_repl_sentence_embedding,
                                                test_origin_sentence_embedding,
                                                test_repl_sentence_embedding,
                                                val_origin_sentence_embedding,
                                                val_repl_sentence_embedding,
                                                train_label, test_label, val_label,
                                                train_sen_similarity,
                                                train_sen_similarity_t,
                                                test_sen_similarity,
                                                test_sen_similarity_t,
                                                val_sen_similarity,
                                                val_sen_similarity_t,
                                                train_nf, test_nf, val_nf,
                                                )
    print(metric_name)
    print(score)
    l = len(predicted_label)
    for i in range(l):
        if predicted_label[i] > 0.5:
            predicted_label[i] = 1
        else:
            predicted_label[i] = 0
    print(predicted_label[0:10])
    # evaluate
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_label, predicted_label))
    # confusion matrix
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(test_label, predicted_label)
    print(cm)
