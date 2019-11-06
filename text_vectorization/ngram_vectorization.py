"""
Authors : Kumarage Tharindu & Paras Sheth
Class : CSE 472
Organization : ASU CIDSE
Project : SMM Project 2
Task : Text preprocessor : Cleaning and pre-processing news data

"""
from nltk import pos_tag
from text_preprocess import preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class Ngram:
    def __init__(self, pos=False, ngram=False):
        self.pos = pos
        self.ngram = ngram

    def fit(self, data, k=1):
        if self.pos:
            data = self.pos_tagger(data)
        if self.ngram:
            features = self.ngrams(data, k)
        return features

    def ngrams(self, filtered_sentence, k=1):
        # TF-IDF to convert text to vectors
        tfidf_vector = TfidfVectorizer(ngram_range=(k, k))
        p = preprocessor.PreProcess()
        features = tfidf_vector.fit_transform(filtered_sentence)
        tf_idf_df = pd.DataFrame(
            features.todense(),
            columns=tfidf_vector.get_feature_names()
        )
        important_words = tf_idf_df.sum().to_dict()
        return tf_idf_df

    def pos_tagger(self, data):
        p = preprocessor.PreProcess()
        data = p.tokenization(data)
        tagged_words = []
        res = ""
        for i in range(len(data)):
            tagged_words.append(pos_tag(data[i]))

        pos_sentence_list = []
        for article in tagged_words:
            res = ""
            for word in article:
                res += " " + word[1]
            res.lstrip()
            pos_sentence_list.append(res)
        return pos_sentence_list
