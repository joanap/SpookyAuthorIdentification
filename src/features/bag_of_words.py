# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:17:21 2017

@author: ASSG
"""

from sklearn.feature_extraction.text import CountVectorizer
from features.abstract_features import AbstractFeatures

class BagOfWords(AbstractFeatures):
    def __init__(self, Nmin, Nmax, word_pattern, min_df = 1, stopwords = None):
        self.ngram_vect = CountVectorizer(ngram_range=(Nmin, Nmax), token_pattern=word_pattern, 
                              min_df = min_df,
                              stop_words = stopwords)
    
    def prepare(self, input_data):
        self.ngram_vect.fit(input_data)
    
    def cook(self, input_data):
        return self.ngram_vect.transform(input_data)

#%% Testing sandbo
#runfile('C:/Users/ASSG/Dropbox/Documents/GitHub/SpookyAuthorIdentification/src2/load_DataFrames.py', wdir='C:/Users/ASSG/Dropbox/Documents/GitHub/SpookyAuthorIdentification/src2')

if __name__ == '__main__':
    #runfile('C:/Users/ASSG/Dropbox/Documents/GitHub/SpookyAuthorIdentification/src2/load_DataFrames.py', wdir='C:/Users/ASSG/Dropbox/Documents/GitHub/SpookyAuthorIdentification/src2')
    testvar = BagOfWords(1,2,r'\b\w\w+\b')
    testvar.prepare(train_DataFrame.text)
    transform_result_bigram  = testvar.cook(train_DataFrame.text)
#    testvar2 = WordBalancer()
#    testvar2.prepare(transform_result_bigram)
#    testvar3 = testvar2.cook(transform_result_bigram)
    