"""
Authors : Kumarage Tharindu & Paras Sheth
Class : CSE 472
Organization : ASU CIDSE
Project : SMM Project 2
Task : Text vectorization : Create a vectorized data set from text corpus based on word embeddings by Word2Vec

"""

from gensim.models import KeyedVectors
from text_preprocess import data_wrapper as dt
import nltk
import numpy as np


class Word2Vec:
    def __init__(self, pre_train_model='GoogleNews-vectors-negative300.bin.gz',
                 embedding_dim=300, aggregate_strategy='sum'):
        self.pre_train_model = pre_train_model
        self.embedding_file_path = dt.get_google_word2vec_path(pre_train_model)
        self.embedding_dim = embedding_dim
        self.aggregate_strategy = aggregate_strategy

    def tokenization(self, data):
        tokens = []
        for i in range(len(data)):
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            tokens.append(tokenizer.tokenize(data[i]))
        return tokens

    def fit(self, data):
        list_of_words = self.tokenization(data)
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_file_path, binary=True)

        data_embedded = []

        if self.aggregate_strategy == 'sum':
            for sentence in list_of_words:
                sum_of_vectors = np.zeros(self.embedding_dim)
                for word in sentence:
                    try:
                        sum_of_vectors += word2vec[word]
                    except:
                        pass
                data_embedded.append(sum_of_vectors)

        if self.aggregate_strategy == 'mean':
            for sentence in list_of_words:
                sum_of_vectors = np.zeros(self.embedding_dim)
                for word in sentence:
                    try:
                        sum_of_vectors += word2vec[word]
                    except:
                        pass
                data_embedded.append(sum_of_vectors/len(sentence))

        return data_embedded

