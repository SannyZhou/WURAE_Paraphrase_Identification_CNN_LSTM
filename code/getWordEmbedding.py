# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Project: paraphrase_detection
# @File: getWordEmbedding.py

from gensim.models import word2vec

# load initialized 300-dimensional word embedding vector
def get_word_embedding():
    print("Build word embedding.")
    try:
        word_vectors = word2vec.Word2VecKeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary = True)
        print("Finish building word embedding.")
    except Exception as e:
        print("Fail in building word embedding",e)
    return word_vectors


def get_word_chinese_embedding():
    print("Building word embedding...")
    try:
        word_vectors = word2vec.Word2VecKeyedVectors.load_word2vec_format('../data/chinese_wordembedding_with_wiki_word2vec_format.bin', binary=True)
        print("Finish building word embedding.")
    except Exception as e:
        print("Fail in building word embedding", e)
    return word_vectors