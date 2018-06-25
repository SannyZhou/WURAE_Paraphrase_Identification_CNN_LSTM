# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: TestRecursiveAutoEncoderTorch.py

import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from function import *
from tree import *
import os, sys, shutil, time, itertools
# import matplotlib.pyplot as plt
import time
from getWordEmbedding import *
from convertStandfordParserTrees import *
import argparse
from collections import Counter
import logging

SAVE_DIR = './torchweights/'
ifcuda = 1


class RecursiveAutoEncoderTorch(nn.Module):
    def __init__(self, embedding_size, lamda_reg=0.15, hidden_size=None,
                 lr=0.01, model_name='raetorch', is_unfold=0,
                 max_epoches=2, anneal_threshold=0.99, anneal_by=1.5,
                 early_stopping=10, batch_size=5000, p_normalize=1):
        super(RecursiveAutoEncoderTorch, self).__init__()
        self.lr = lr
        self.lamda_reg = lamda_reg
        self.model_name = model_name
        self.is_unfold = is_unfold
        self.max_epoches = max_epoches
        self.anneal_threshold = anneal_threshold
        self.anneal_by = anneal_by
        self.early_stopping = early_stopping
        self.batch_size = batch_size

        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.p_normalize = p_normalize
        self.encoder = nn.Linear(self.embedding_size * 2, self.hidden_size).double()
        self.decoder_1 = nn.Linear(self.hidden_size, self.embedding_size).double()
        self.decoder_2 = nn.Linear(self.hidden_size, self.embedding_size).double()
        self.loss = 0.0
        # self.optimizer = optim.LBFGS(self.parameters(), lr=self.lr, max_iter=20, max_eval=None, tolerance_grad=1e-05,
        #                          tolerance_change=1e-09, history_size=100, line_search_fn=None)
        self.double()

    def encode(self, c1, c2):
        input_c = torch.cat((c1, c2), dim=0)
        f = nn.Tanh()
        out_p_z = self.encoder(input_c)
        p_unnormalized = f(out_p_z)
        if self.p_normalize:
            p = p_unnormalized / torch.norm(p_unnormalized, p=2)
        else:
            p = p_unnormalized
        return p, p_unnormalized

    def predict_forward(self, word_embedding_matrix, sKids):
        sentence_len = word_embedding_matrix.shape[0]
        t = Tree_torch()

        t.nodeFeatures = np.concatenate([word_embedding_matrix,
                                                      np.zeros([sentence_len - 1, self.embedding_size])],
                                                     axis=0)
        t.nodeFeatures_unnormalized = np.concatenate([word_embedding_matrix,
                                         np.zeros([sentence_len - 1, self.embedding_size])],
                                        axis=0)

        # encoding part and decoding part
        for index in range(sentence_len, 2 * sentence_len - 1):
            kids = sKids[index]
            # get the word embedding of child nodes
            c1 = t.nodeFeatures[kids[0] - 1]
            c2 = t.nodeFeatures[kids[1] - 1]
            c1 = torch.from_numpy(c1)
            c2 = torch.from_numpy(c2)
            if ifcuda:
                c1 = Variable(c1).cuda()
                c2 = Variable(c2).cuda()
            else:
                c1 = Variable(c1)
                c2 = Variable(c2)
            # encoding via parameters
            p, p_unnormalized = self.encode(c1, c2)
            if ifcuda:
                t.nodeFeatures_unnormalized[index] = p_unnormalized.data.cpu().numpy()
                t.nodeFeatures[index] = p.data.cpu().numpy()
            else:
                t.nodeFeatures_unnormalized[index] = p_unnormalized.data.numpy()
                t.nodeFeatures[index] = p.data.numpy()

        return t.nodeFeatures[sentence_len:]


def predict_result(rae, instances, embedding_size):
    num = len(instances['allSStr'])
    nodefeature_list = []
    nodefeature_sen1_sen2 = []
    for n in range(num):
        word_embedding_sentence = []
        for word in instances['allSStr'][n]:
            if word in wordvec:
                word_embedding_sentence.append(np.array(wordvec[word]))
            else:
                word_embedding_sentence.append((np.random.rand(300) / 50 - 0.01))

        word_embedding_sentence = np.array(word_embedding_sentence)
        # forward propagation
        nodefeature = rae.predict_forward(word_embedding_sentence, instances['allSKids'][n])
        if n % 2 == 1:
            nodefeature_sen1_sen2.append(nodefeature)
            nodefeature_list.append(nodefeature_sen1_sen2)
            nodefeature_sen1_sen2 = []
        else:
            nodefeature_sen1_sen2.append(nodefeature)
    return nodefeature_list


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    hidden_size = 300
    deep_layers = 0
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model-weight', required=True,
                            help='model name')
    arg_parser.add_argument('-gpu', type=int, default=1)
    arg_parser.add_argument('-parsed', required=True)

    options = arg_parser.parse_args()

    # instances_file = options.instances
    model_weight = options.model_weight
    ifcuda = options.gpu
    train_parsed = options.parsed

    print('Building model...')
    logging.info('Building model...')
    rae = RecursiveAutoEncoderTorch(300)

    if ifcuda:
        rae.cuda()

    print('Converting parsed line to tree...')
    train_parsed_tree = convertStanfordParserTrees(train_parsed)

    print('Loading word embedding corpus...')
    logging.info('Loading word embedding corpus...')
    global wordvec
    wordvec = get_word_embedding()
    # embedding_size = wordvec['test'].shape[0]
    # print('The size of embedding is', embedding_size)
    # embedding_size = 300

    rae.load_state_dict(torch.load(model_weight))

    print('Predicting...')
    nodefeature_list = predict_result(rae, train_parsed_tree, embedding_size=300)
    print(nodefeature_list[0:5])
    nodefeature_file = model_weight.split('.')[0] + '_' + train_parsed.split('/')[-1].split('.')[-2] + '_nodeFeature' + '.pickle'

    print('Saving node feature to %s' % nodefeature_file)
    nf = open(nodefeature_file, 'wb')
    pickle.dump(nodefeature_list, nf)

    print('Done!')

