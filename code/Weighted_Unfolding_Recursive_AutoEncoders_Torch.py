# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: Weighted_Unfolding_Recursive_AutoEncoders_Torch.py

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
import pickle

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

    def decode(self, p):
        f = nn.Tanh()
        y1_unnormalized = f(self.decoder_1(p))
        y2_unnormalized = f(self.decoder_2(p))
        y1 = y1_unnormalized / torch.norm(y1_unnormalized, p=2)
        y2 = y2_unnormalized / torch.norm(y2_unnormalized, p=2)
        return y1, y2, y1_unnormalized, y2_unnormalized

    def forward(self, word_embedding_matrix, sKids, sCounts=None, predict_forward=0):
        # forward propagation
        sentence_len = word_embedding_matrix.shape[0]
        t = Tree_torch()
        t.nodeFeatures = np.concatenate([word_embedding_matrix,
                                         np.zeros([sentence_len - 1, self.embedding_size])],
                                        axis=0)
        t.kids = sKids
        t.counts = sCounts
        J = 0.0
        # encoding part and decoding part
        for index in range(sentence_len, 2 * sentence_len - 1):
            kids = sKids[index]
            # get the word embedding of child nodes
            c1 = t.nodeFeatures[kids[0] - 1]
            c2 = t.nodeFeatures[kids[1] - 1]
            if ifcuda:
                c1 = Variable(torch.from_numpy(c1)).cuda()
                c2 = Variable(torch.from_numpy(c2)).cuda()
            else:
                c1 = Variable(torch.from_numpy(c1))
                c2 = Variable(torch.from_numpy(c2))
            # encoding via parameters
            p, p_unnormalized = self.encode(c1, c2)
            if ifcuda:
                t.nodeFeatures[index] = p.data.cpu().numpy()
                # t.nodeFeatures_unnormalized[index] = p_unnormalized.data.cpu().numpy()
            else:
                t.nodeFeatures[index] = p.data.numpy()
                # t.nodeFeatures_unnormalized[index] = p_unnormalized.data.numpy()
            # decoding part and calculate reconstruction error
            if self.is_unfold:
                tmp_j = self.decoding_part_unfolding(t, p, index, sentence_len, predict_forward)
            else:
                tmp_j = self.decoding_part(t, p, index)
            J = J + tmp_j
        return J

    def decoding_part(self, t, p, index):
        kids = t.kids[index]
        y1, y2, y1_unnormalized, y2_unnormalized = self.decode(p)
        if ifcuda:
            c1_var = Variable(torch.from_numpy(t.nodeFeatures[kids[0] - 1])).cuda()
            c2_var = Variable(torch.from_numpy(t.nodeFeatures[kids[1] - 1])).cuda()
        else:
            c1_var = Variable(torch.from_numpy(t.nodeFeatures[kids[0] - 1]))
            c2_var = Variable(torch.from_numpy(t.nodeFeatures[kids[1] - 1]))
        err = (y1 - c1_var).pow(2).sum() + (y2 - c2_var).pow(2).sum()
        err = 0.5 * err
        return err

    def decoding_part_unfolding(self, t, p, index, sentence_len, predict_forward=0):
        kids = t.kids[index]
        y1, y2, y1_unnormalized, y2_unnormalized = self.decode(p)
        err = 0.0
        if kids[0] <= sentence_len:
            if ifcuda:
                c_var = Variable(torch.from_numpy(t.nodeFeatures[kids[0] - 1])).cuda()
            else:
                c_var = Variable(torch.from_numpy(t.nodeFeatures[kids[0] - 1]))
            if ifweighted and not predict_forward:
                err += 1.0 / t.counts[kids[0] - 1] * 0.5 * (y1 - c_var).pow(2).sum()
            else:
                err += 0.5 * (y1 - c_var).pow(2).sum()
        else:
            err += self.decoding_part_unfolding(t, y1, kids[0] - 1, sentence_len, predict_forward)
        if kids[1] <= sentence_len:
            if ifcuda:
                c_var = Variable(torch.from_numpy(t.nodeFeatures[kids[1] - 1])).cuda()
            else:
                c_var = Variable(torch.from_numpy(t.nodeFeatures[kids[1] - 1]))
            if ifweighted and not predict_forward:
                err += 1.0 / t.counts[kids[1] - 1] * 0.5 * (y2 - c_var).pow(2).sum()
            else:
                err += 0.5 * (y2 - c_var).pow(2).sum()
        else:
            err += self.decoding_part_unfolding(t, y2, kids[1] - 1, sentence_len, predict_forward)
        return err

    def predict_forward(self, word_embedding_matrix, sKids):
        sentence_len = word_embedding_matrix.shape[0]
        t = Tree_torch()
        t.nodeFeatures = np.concatenate([word_embedding_matrix,
                                         np.zeros([sentence_len - 1, embedding_size])],
                                        axis=0)
        t.nodeFeatures_unnormalized = np.concatenate([word_embedding_matrix,
                                                      np.zeros([sentence_len - 1, embedding_size])],
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
                t.nodeFeatures[index] = p.data.cpu().numpy()
                t.nodeFeatures_unnormalized[index] = p_unnormalized.data.cpu().numpy()
            else:
                t.nodeFeatures[index] = p.data.numpy()
                t.nodeFeatures_unnormalized[index] = p_unnormalized.data.numpy()

        return t.nodeFeatures_unnormalized


# def plot_loss_history_epoch(loss):
#     loss_file = open(rae.model_name + 'loss_history_epoch', 'wb')
#     pickle.dump(loss, loss_file)
#     loss_file.close()
#     # plt.plot(loss)
#     # plt.title('Loss history')
#     # plt.xlabel('Iteration')
#     # plt.ylabel('Loss')
#     # plt.savefig(rae.model_name+'loss_history.png')
#     return


def plot_loss_history(stats):
    loss_file = open(rae.model_name + 'loss_history_final', 'wb')
    pickle.dump(stats['loss_history'], loss_file)
    loss_file.close()
    # plt.plot(stats['loss_history'])
    # plt.title('Loss history')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.savefig(rae.model_name+'loss_history_final.png')
    return


def batch_iter(kids, strs, count=None, batch_size=64):
    # data with batch size
    data_len = len(kids)

    indices = np.random.permutation(np.arange(data_len))
    kids_shuffle = np.array(kids)[indices]
    str_shuffle = np.array(strs)[indices]
    if ifweighted:
        count_shuffle = np.array(count)[indices]
        return kids_shuffle, str_shuffle, count_shuffle
    else:
        return kids_shuffle, str_shuffle


def get_time_dif(start_time):
    # get the time used
    end_time = time.time()
    time_dif = end_time - start_time
    return int(round(time_dif))


def run_epoch(val_data, train_data, epoch, best_val_loss=float('inf')):
    loss_history = []
    allskids = train_data['allSKids']
    allsstr = train_data['allSStr']
    mini_mean_loss = float('inf')
    if ifweighted:
        allscount = train_data['allSCounter']
        newkidslist, newstrlist, newcounterlist = batch_iter(allskids, allsstr, allscount)
    else:
        newkidslist, newstrlist = batch_iter(allskids, allsstr)
    if ifweighted:
        del allscount, allskids, allsstr
    else:
        del allskids, allsstr
    num = len(newstrlist)
    num_batch = int((num - 1) / rae.batch_size) + 1
    start_time = time.time()
    stopped = -1
    best_val_batch = 0
    for i in range(num_batch):
        print('batch %d:' % i)
        logging.info('batch '+str(i)+':')
        start_id = i * rae.batch_size
        end_id = min((i + 1) * rae.batch_size, num)
        kids = newkidslist[start_id:end_id]
        strs = newstrlist[start_id:end_id]
        if ifweighted:
            counts = newcounterlist[start_id:end_id]
        batch_num_nodes = 0
        for str_each in strs:
            batch_num_nodes += len(str_each) - 1

        def closure():
            optimizer.zero_grad()
            if ifweighted:
                loss = run_batch(kids, strs, batch_num_nodes, counts)
            else:
                loss = run_batch(kids, strs, batch_num_nodes)
            if ifcuda:
                loss_history.append(loss.data.cpu().numpy()[0])
            else:
                loss_history.append(loss.data.numpy()[0])
            l2_loss = calculate_l2()
            l2_loss.backward()
            batch_loss = l2_loss + loss
            return batch_loss

        optimizer.step(closure)
        print('\r{} / {}:   loss = {}'.format(end_id, num, np.mean(loss_history)), end='\t')
        logging.info('\t'+str(end_id)+'/'+str(num)+':  loss = '+str(np.mean(loss_history))+'\t')
        print('Time: ', get_time_dif(start_time), end=' ')
        logging.info('Time'+str(get_time_dif(start_time))+' ')
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        torch.save(rae.state_dict(), SAVE_DIR+'%stmp.pkl' % rae.model_name)
        # plot_loss_history_batch(loss_history)
        # if np.mean(loss_history) > mini_mean_loss:
        #     rae.lr /= (rae.anneal_by * 0.9)
        # else:
        #     mini_mean_loss = np.mean(loss_history)
        if i % 2 == 0 or i == num_batch - 1:
            if np.mean(loss_history) > mini_mean_loss:
                rae.lr /= (rae.anneal_by * 0.9)
            else:
                mini_mean_loss = np.mean(loss_history)
            val_loss = predict(val_data)
            print('Val_loss = ', val_loss)
            logging.info('Val_loss = ' + str(val_loss) + '')
            if best_val_loss > val_loss:
                shutil.copyfile(SAVE_DIR + '%stmp.pkl' % rae.model_name,
                                SAVE_DIR + '%s%d%d.pkl' % (rae.model_name, epoch, i))
                best_val_loss = val_loss
                best_val_batch = i
                print('*')
                logging.info('*')
            else:
                rae.lr /= (rae.anneal_by * 0.9)
                print('annealed learning rate to %f' % rae.lr)
                logging.info('annealed learning rate to' + str(rae.lr) + '')
                print(' ')
                logging.info(' ')
            # if model hasn't improved for a while stop
            if i - best_val_batch > rae.early_stopping:
                stopped = i
                break
        else:
            print(' ')
            logging.info(' ')
    print('\n\nstopped at %d batch\n' % stopped)
    logging.info('\n\nstopped at'+str(stopped)+'batch\n')
    return loss_history, best_val_loss, stopped


def calculate_l2():
    l2_reg = None
    for W in rae.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg * rae.lamda_reg


def run_batch(kids, strs, num_nodes, counts=None):
    num = len(strs)
    batch_cost = 0.0
    for n in range(num):
        # print(n)
        word_embedding_matrix = []
        for word in strs[n]:
            if word in wordvec:
                word_embedding_matrix.append(wordvec[word])
            else:
                word_embedding_matrix.append((np.random.rand(300) / 50 - 0.01))
        word_embedding_matrix = np.array(word_embedding_matrix)
        # forward propatation
        if ifweighted:
            rec_err = rae.forward(word_embedding_matrix, kids[n], counts[n])
        else:
            rec_err = rae.forward(word_embedding_matrix, kids[n])
        rec_err = rec_err / num_nodes
        rec_err.backward()
        batch_cost += rec_err
    return batch_cost


def train_torch(train_instances, val_instances, verbose=True):
    complete_loss_history = []
    prev_epoch_loss = float('inf')
    best_val_loss = float('inf')
    # best_val_epoch = 0
    # stopped = -1
    start_time = time.time()
    for epoch in range(rae.max_epoches):
        print('Epoch %d:' % epoch)
        logging.info('Epoch '+str(epoch)+':')
        loss_history, val_loss, stopped_batch = run_epoch(val_instances, train_instances, epoch,
                                                            best_val_loss=best_val_loss)
        complete_loss_history.extend(loss_history)
        # plot_loss_history_epoch(complete_loss_history)
        # annealing learning rate
        epoch_loss = np.mean(loss_history)
        if epoch_loss > prev_epoch_loss * rae.anneal_threshold:
            rae.lr /= rae.anneal_by
            print('annealed learning rate to %f' % rae.lr)
            logging.info('annealed learning rate to' + str(rae.lr)+'')
        prev_epoch_loss = epoch_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    print('Training time: ', get_time_dif(start_time))
    logging.info('Training Time: '+str(get_time_dif(start_time))+'')
    return {
        'loss_history': complete_loss_history,
    }


def predict(instances):
    str_list = instances['allSStr']
    kids = instances['allSKids']
    cost = 0.0
    num = len(str_list)
    pre_batch_size = 500
    num_batch = int((num - 1) / pre_batch_size) + 1
    num_nodes = 0
    for str_each in str_list:
        num_nodes += len(str_each) - 1
    for pre_i in range(num_batch):
        # print('batch %d:' % pre_i)
        start_id = pre_i * pre_batch_size
        end_id = min((pre_i + 1) * pre_batch_size, num)
        kids_batch = kids[start_id:end_id]
        str_batch = str_list[start_id:end_id]
        cost += predict_batch(str_batch, kids_batch)
    return cost/num_nodes


def predict_batch(str_batch, kids):
    num = len(str_batch)
    cost_predict_batch = 0.0
    for n in range(num):
        word_embedding_matrix = []
        for word in str_batch[n]:
            if word in wordvec:
                word_embedding_matrix.append(wordvec[word])
            else:
                word_embedding_matrix.append((np.random.rand(300) / 50 - 0.01))
        word_embedding_matrix = np.array(word_embedding_matrix, dtype=np.float64)
        # forward propatation
        rec_err = rae.forward(word_embedding_matrix, kids[n], predict_forward=1)
        cost_predict_batch += rec_err
    if ifcuda:
        return cost_predict_batch.data.cpu().numpy()[0]
    else:
        return cost_predict_batch.data.numpy()[0]


def add_count(trees):
    check_count_list = []
    for str_list in trees['allSStr']:
        check_count = []
        for check_word in str_list:
            if check_word in english_counter:
                check_count.append(english_counter[check_word])
            else:
                check_count.append(1)
        check_count_list.append(check_count)

    new_trees = {'allSStr': trees['allSStr'], 'allSKids': trees['allSKids'], 'allSCounter': check_count_list}
    return new_trees


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
    hidden_size = 300
    deep_layers = 0
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--hidden-size',
                            help='size in the deep layer')
    arg_parser.add_argument('--deep-layers',
                            help='the number of deep layers in the encoding or decoding part')
    arg_parser.add_argument('-model', required=True,
                            help='model name')
    arg_parser.add_argument('-lambda_reg', type=float, default=0.125,
                            help='weight of the regularizer')

    arg_parser.add_argument('--batch-size', type=int, default=3000,
                            help='the size of batch while training')
    arg_parser.add_argument('-lr', type=float, default=0.01,
                            help='learning rate for gradient descent optimizer')
    arg_parser.add_argument('-epoch', type=int, default=30,
                            help='max epoch for training')
    arg_parser.add_argument('-athreshold', type=float, default=0.99,
                            help='anneal threshold')
    arg_parser.add_argument('-aby', type=float, default=1.5,
                            help='anneal the learning rate')
    arg_parser.add_argument('--early-stopping', type=int, default=4)
    arg_parser.add_argument('-unfold', type=int, default=0)
    arg_parser.add_argument('-gpu', type=int, default=1)
    arg_parser.add_argument('--embedding-size', type=int, default=300)
    arg_parser.add_argument('-weighted', type=int, default=0)

    options = arg_parser.parse_args()

    model = options.model
    lambda_reg = options.lambda_reg
    lr = options.lr
    athreshold = options.athreshold
    aby = options.aby
    epoch = options.epoch
    batch_size = options.batch_size
    early_stopping = options.early_stopping
    is_unfold = options.unfold
    ifcuda = options.gpu
    embedding_size = options.embedding_size
    ifweighted = options.weighted

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s',
                        datefmt='%Y-%m-%d(%a)%H:%M:%S',
                        filename=model+'_output.txt',
                        filemode='a')

    # train_parsed = '../data/sentence_parsed.txt'
    # val_parsed = '../data/sentence_msr_paraphrase_testparsed.txt'
    train_parsed = '../data/sentence_msr_paraphrase_trainparsed.txt'
    val_parsed = '../data/sentence_msr_paraphrase_testparsed.txt'
    print('Building model...')
    logging.info('Building model...')
    rae = RecursiveAutoEncoderTorch(embedding_size, lambda_reg, lr=lr, batch_size=batch_size, max_epoches=epoch,
                                    anneal_threshold=athreshold, anneal_by=aby, hidden_size=None, model_name=model,
                                    early_stopping=early_stopping, is_unfold=is_unfold)
    rae.double()
    # rae.load_state_dict(torch.load('torweights/english_wurae.pkl'))
    print(rae.lamda_reg, rae.lr, rae.batch_size, rae.max_epoches, rae.anneal_by, rae.anneal_threshold,
          rae.early_stopping, rae.is_unfold)
    if ifcuda:
        rae.cuda()

    os.system('nvidia-smi')
    os.system('ps')

    print('Converting parsed line to tree...')
    logging.info('Converting parsed line to tree...')
    train_parsed_tree = convertStanfordParserTrees(train_parsed)
    val_parsed_tree = convertStanfordParserTrees(val_parsed)

    if ifweighted:
        counter_file = open('../data/english_counter.pkl', 'rb')
        english_counter = pickle.load(counter_file)
        train_parsed_tree = add_count(train_parsed_tree)
        del english_counter

    print('Loading word embedding corpus...')
    logging.info('Loading word embedding corpus...')
    global wordvec
    wordvec = get_word_embedding()
    # embedding_size = wordvec['test'].shape[0]
    print('The size of embedding is', embedding_size)
    logging.info('The size of embedding is' + str(embedding_size) + '')

    print('Training...')
    logging.info('Training...')
    optimizer = optim.LBFGS(rae.parameters(), rae.lr, max_iter=8)
    stats = train_torch(train_parsed_tree, val_parsed_tree, verbose=True)
    plot_loss_history(stats)

    print('Done!')
    logging.info('Done!')