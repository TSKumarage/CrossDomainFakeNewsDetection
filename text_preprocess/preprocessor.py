"""
Authors : Kumarage Tharindu & Paras Sheth
Class : CSE 472
Organization : ASU CIDSE
Project : SMM Project 2
Task : Text preprocessor : Cleaning and pre-processing news data

"""


class PreProcess:
    def __init__(self,special_chars_norm=False, accented_norm=False, contractions_norm=False,
                 stemming_norm=False, lemma_norm=False, stopword_norm=False):
        self.special_chars_norm = special_chars_norm
        self.accented_norm = accented_norm
        self.contractions_norm = contractions_norm
        self.stemming_norm =stemming_norm
        self.lemma_norm = lemma_norm
        self.stopword_norm = stopword_norm

    def special_char_remove(self, data): # Remove special characters
        print("Enter your code")
        return data

    def accented_word_normalization(self, data):  # Normalize accented chars/words
        print("Enter your code")
        return data

    def expand_contractions(self, data):  # Expand contractions
        print("Enter your code")
        return data

    def stemming(self, data):  # Expand contractions
        print("Enter your code")
        return data

    def lemmatization(self, data):  # Expand contractions
        print("Enter your code")
        return data

    def stopword_remove(self, data):  # Remove special characters
        print("Enter your code")
        return data

    def tokenization(self, data):
        print("Enter your code")
        return data

    def fit(self, data):
        if self.contractions_norm:
            data = self.expand_contractions(data)

        if self.accented_norm:
            data = self.accented_word_normalization(data)

        if self.special_chars_norm:
            data = self.special_char_remove(data)

        if self.stemming_norm:
            data = self.stemming(data)

        if self.lemma_norm:
            data = self.lemmatization(data)

        return data

