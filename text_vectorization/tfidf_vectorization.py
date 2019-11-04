"""
Authors : Kumarage Tharindu & Paras Sheth
Class : CSE 472
Organization : ASU CIDSE
Project : SMM Project 2
Task : Text preprocessor : Cleaning and pre-processing news data

"""
from text_preprocess import preprocessor
from text_preprocess import data_wrapper as dt
import numpy as np
import pandas as pd


def term_frequency(data):
    tf = []
    for j in range(len(data)):
        tf.append(data.count(data[j]))
    return tf


def document_frequency(data, word):
    c = 0
    for i in range(len(data)):
        if word in data[i]:
            c += 1
    return c


def tf_idf(data):
    p = preprocessor.PreProcess()
    res = {}
    data = p.tokenization(data)
    for i in range(len(data)):
        tf = term_frequency(data[i])
        df = []
        for j in range(len(data[i])):
            d = document_frequency(data, data[i][j])
            idf = np.log(len(data) / d)
            df.append(idf)
        tf_idf_list = []
        for j in range(len(df)):
            tf_idf_list.append(tf[j] * df[j])
        tf_idf_df = pd.DataFrame(
            [tf_idf_list],
            columns=data[i]
        )
        res[i] = tf_idf_df
    return res
