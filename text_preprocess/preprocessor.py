"""
Authors : Kumarage Tharindu & Paras Sheth
Class : CSE 472
Organization : ASU CIDSE
Project : SMM Project 2
Task : Text preprocessor : Cleaning and pre-processing news data

"""
from text_preprocess import data_wrapper as dt
import nltk
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

class PreProcess:
    def __init__(self,special_chars_norm=False, accented_norm=False, contractions_norm=False,
                 stemming_norm=False, lemma_norm=False, stopword_norm=False, proper_norm=False):
        self.special_chars_norm = special_chars_norm
        self.accented_norm = accented_norm
        self.contractions_norm = contractions_norm
        self.stemming_norm =stemming_norm
        self.lemma_norm = lemma_norm
        self.stopword_norm = stopword_norm
        self.proper_norm = proper_norm

    def special_char_remove(self, data): # Remove special characters
        print("Enter your code")
        return data

    def accented_word_normalization(self, data):  # Normalize accented chars/words
        print("Enter your code")
        return data

    def expand_contractions(self, data):  # Expand contractions
        print("Enter your code")
        return data

    def stemming(self, data):
        stemmer = nltk.stem.PorterStemmer()
        tokens = self.tokenization(data)
        stemmed_data = []
        for i in range(len(tokens)):
            s1 = " ".join(stemmer.stem(tokens[i][j]) for j in range(len(tokens[i])))
            stemmed_data.append(s1)
        return stemmed_data

    def lemmatization(self, data):
        lemma = nltk.stem.WordNetLemmatizer()
        tokens = self.tokenization(data)
        lemmatized_data = []
        for i in range(len(tokens)):
            s1 = " ".join(lemma.lemmatize(tokens[i][j]) for j in range(len(tokens[i])))
            lemmatized_data.append(s1)
        return lemmatized_data

    def stopword_remove(self, data):  # Remove special characters
        filtered_sentence = []
        stop_words = set(stopwords.words('english'))
        data = self.tokenization(data)
        for i in range(len(data)):
            res = ""
            for j in range(len(data[i])):
                if data[i][j].lower() not in stop_words:
                    res = res + " " + data[i][j]
            filtered_sentence.append(res)
        return filtered_sentence

    def remove_proper_nouns(self, data):
        common_words = []
        data = self.tokenization(data)
        for i in range(len(data)):
            tagged_sent = pos_tag(data[i])
            proper_nouns = [word for word, pos in tagged_sent if pos == 'NNP']
            res = ""
            for j in range(len(data[i])):
                if data[i][j] not in proper_nouns:
                    res = res + " " + data[i][j]
            common_words.append(res)
        return common_words

    def tokenization(self, data):
        tokens = []
        for i in range(len(data)):
            tokenizer = nltk.tokenize.TreebankWordTokenizer()
            tokens.append(tokenizer.tokenize(data[i]))
        return tokens

    def fit(self, data):
        if self.contractions_norm:
            data = self.expand_contractions(data)

        if self.accented_norm:
            data = self.accented_word_normalization(data)

        if self.special_chars_norm:
            data = self.special_char_remove(data)

        if self.stemming_norm:
            data = self.stemming(data)

        if self.proper_norm:
            data = self.remove_proper_nouns(data)

        if self.stopword_norm:
            data = self.stopword_remove(data)

        if self.lemma_norm:
            data = self.lemmatization(data)

        return data


def main():
    data = dt.read_data('gossipcop_content_no_ignore',type="numpy")
    new_data = data[:, 2]
    p = PreProcess(lemma_norm=True, proper_norm=True, stopword_norm=True)
    pre_processed_data = p.fit(new_data)
    print(pre_processed_data[0])



if __name__=='__main__':
    main()
