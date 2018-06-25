# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: fileprocess.py

from nltk.tokenize import RegexpTokenizer
import pickle
import os


def msrp_data(filename):
    label, sid, s1, s2 = [], [], [], []
    sentences = ''
    f = open(filename, 'r')
    f.readline()
    tokenizer = RegexpTokenizer(r'\w+')
    for l in f.readlines():
        line = l.strip().split('\t')
        label.append(int(line[0]))
        sid.append([line[1], line[2]])
        sentences += line[3] + '\n'
        sentences += line[4] + '\n'
        s1.append(tokenizer.tokenize(line[3].lower()))
        s2.append(tokenizer.tokenize(line[4].lower()))
    f.close()
    df = {'label': label,
          'sid': sid,
          's1': s1,
          's2': s2}
    sentence_file = 'sentence_' + filename.split('/')[-1]
    f = open(sentence_file, 'w')
    f.write(sentences)
    f.close()
    parsed_file = sentence_to_parse_tree(sentence_file)
    return df, parsed_file


def sentence_to_parse_tree(sentencefilename):
    parsed_file = sentencefilename.split('.')[0] + 'parsed' + '.' + sentencefilename.split('.')[1]
    os.system('../../parser/stanford-parser-full-2018-02-27/lexparser.sh ' + sentencefilename + ' > ' + parsed_file)
    return parsed_file

# if __name__ == '__main__':
#     test_file = 'msr_paraphrase_test.txt'
#     _, sentence_test_file = msrp_data(test_file)
#     print(sentence_test_file)