# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: new_features.py

import sys,os
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import math
import nltk
import json
from nltk.corpus import wordnet as wn


# get LCS(longest common subsquence),DP
def lcs(str_a, str_b):
    lensum = float(len(str_a) + len(str_b))
    # dp[lena+1][lenb+1] and initialize to 0
    lengths = [[0 for j in range(len(str_b) + 1)] for i in range(len(str_a) + 1)]
    # enumerate(a)函数： 得到下标i和a[i]
    for i, x in enumerate(str_a):
        for j, y in enumerate(str_b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # after getting the longest length of LCS, then get the LCS from the sequence
    result = ""
    x, y = len(str_a), len(str_b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert str_a[x - 1] == str_b[y - 1]
            result = str_a[x - 1] + result
            x -= 1
            y -= 1
    longestdist = lengths[len(str_a)][len(str_b)]
    ratio = longestdist / min(len(str_a), len(str_b))
    # return {'longestdistance':longestdist, 'ratio':ratio, 'result':result}
    return ratio


def minimumEditDistance(str_a, str_b):
    lensum = float(len(str_a) + len(str_b))
    if len(str_a) > len(str_b):
        str_a, str_b = str_b, str_a
    distances = range(len(str_a) + 1)
    for index2, char2 in enumerate(str_b):
        # str_b > str_a
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(str_a):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],  # delete
                                             distances[index1 + 1],  # insert
                                             newDistances[-1])))  # exchange
        distances = newDistances
    mindist = distances[-1]
    ratio = (lensum - mindist) / lensum
    return ratio


def get_sentence_bleu(reference_list, hypothesis):
    reference_list = [reference_list]
    weights = [1]
    sentence_bleu_score_1 = nltk.translate.bleu_score.sentence_bleu(reference_list, hypothesis, weights)
    weights = [0.5, 0.5]
    sentence_bleu_score_2 = nltk.translate.bleu_score.sentence_bleu(reference_list, hypothesis, weights)
    weights = [0.333, 0.333, 0.333]
    sentence_bleu_score_3 = nltk.translate.bleu_score.sentence_bleu(reference_list, hypothesis, weights)
    weights = [0.25, 0.25, 0.25, 0.25]
    sentence_bleu_score_4 = nltk.translate.bleu_score.sentence_bleu(reference_list, hypothesis, weights)
    return sentence_bleu_score_1, sentence_bleu_score_2, sentence_bleu_score_3, sentence_bleu_score_4


def eng_tiidf_feature(str_a, str_b):
    f = open('../data/en.json', 'r')
    stop_words = json.load(f)
    f.close()
    stop_words = set(stop_words)
    cropus = [str_a, str_b]
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(tfidf_vectorizer.fit_transform(cropus))
    tfidf = tfidf.toarray()
    n = float(np.dot(tfidf[0], tfidf[1].T))
    denom = np.linalg.norm(tfidf[0]) * np.linalg.norm(tfidf[1])
    if denom == 0.0:
        sim = 0
    else:
        cos = n / denom
        sim = 0.5 + 0.5 * cos
    return sim


def chinese_tiidf_feature(str_a, str_b):
    f = open('../data/stopwords.dat', 'r')
    stop_words = f.readlines()
    f.close()
    stop_words = set(stop_words)
    cropus = [str_a, str_b]
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(tfidf_vectorizer.fit_transform(cropus))
    tfidf = tfidf.toarray()
    n = float(np.dot(tfidf[0], tfidf[1].T))
    denom = np.linalg.norm(tfidf[0]) * np.linalg.norm(tfidf[1])
    if denom == 0.0:
        sim = 0
    else:
        cos = n / denom
        sim = 0.5 + 0.5 * cos
    return sim


