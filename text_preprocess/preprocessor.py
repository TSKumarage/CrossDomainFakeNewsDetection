"""
Authors : Kumarage Tharindu & Paras Sheth
Class : CSE 472
Organization : ASU CIDSE
Project : SMM Project 2
Task : Text preprocessor : Cleaning and pre-processing news data

"""
from text_preprocess import data_wrapper as dt
from text_preprocess import contractions as ct
import nltk
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import unicodedata
import re
from pycontractions import Contractions


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

    def special_char_remove(self, data, remove_digits=False): # Remove special characters
        tokens = self.tokenization(data)
        special_char_norm_data = []

        for token in tokens:
            sentence = ""
            for word in token:
                sentence += word + " "
            sentence.rstrip()

            clean_remove = re.compile('<.*?>')
            norm_sentence = re.sub(clean_remove, '', sentence)
            norm_sentence = norm_sentence.replace(".", "")
            norm_sentence = norm_sentence.replace("\\", "")
            norm_sentence = norm_sentence.replace("-", " ")
            norm_sentence = norm_sentence.replace(",", "")
            special_char_norm_data.append(norm_sentence)

        return special_char_norm_data

    def accented_word_normalization(self, data):  # Normalize accented chars/words
        tokens = self.tokenization(data)
        accented_norm_data = []

        for token in tokens:
            sentence = ""
            for word in token:
                sentence += word + " "
            sentence.rstrip()
            norm_sentence = unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8', 'ignore')

            accented_norm_data.append(norm_sentence)

        return accented_norm_data

    def expand_contractions(self, data, pycontrct=True):  # Expand contractions

        if pycontrct:  # Contraction removal based on Google news word2vec
            tokens = self.tokenization(data)
            cont = Contractions(dt.get_google_word2vec_path())
            contraction_norm_data = cont.expand_texts(data, precise=True)
            return contraction_norm_data

        else:   # Simple contraction removal based on pre-defined set of contractions
            contraction_mapping = ct.CONTRACTION_MAP
            contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                              flags=re.IGNORECASE | re.DOTALL)

            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expanded_contraction = contraction_mapping.get(match) \
                    if contraction_mapping.get(match) \
                    else contraction_mapping.get(match.lower())
                expanded_contraction = first_char + expanded_contraction[1:]
                return expanded_contraction

            tokens = self.tokenization(data)
            contraction_norm_data = []

            for token in tokens:
                sentence = ""
                for word in token:
                    sentence += word + " "
                sentence.rstrip()

                expanded_text = contractions_pattern.sub(expand_match, sentence)
                expanded_text = re.sub("'", "", expanded_text)

                contraction_norm_data.append(expanded_text)

            return contraction_norm_data

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
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            tokens.append(tokenizer.tokenize(data[i]))

        return tokens

    def fit(self, data):
        if self.contractions_norm:
            data = self.expand_contractions(data)

        if self.accented_norm:
            data = self.accented_word_normalization(data)

        if self.special_chars_norm:
            data = self.special_char_remove(data, remove_digits=False)

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
    new_data = data[0:1, 2]
    print(new_data)
    print()
    p = PreProcess(lemma_norm=False, proper_norm=False, stopword_norm=False,
                   accented_norm=True, special_chars_norm=True, contractions_norm=True)
    pre_processed_data = p.fit(new_data)
    print(pre_processed_data[0])


if __name__ == '__main__':
    main()
