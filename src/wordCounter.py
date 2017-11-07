# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:26:52 2017

@author: Joana Pinto
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:50:11 2017

@author: Joana Pinto
"""

#%% import libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#default constructor
#CountVectorizer(analyzer=...'word', binary=False, decode_error=...'strict',
#        dtype=<... 'numpy.int64'>, encoding=...'utf-8', input=...'content',
#        lowercase=True, max_df=1.0, max_features=None, min_df=1,
#        ngram_range=(1, 1), preprocessor=None, stop_words=None,
#        strip_accents=None, token_pattern=...'(?u)\\b\\w\\w+\\b',
#        tokenizer=None, vocabulary=None)

unigram_vect = CountVectorizer()

bigram_vect = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

train_DataFrame = pd.read_csv('../../input/train.csv')
train_DataFrame.describe()

#get word occurences per sentence
unigram_counts = unigram_vect.fit_transform(train_DataFrame['text'])
bigram_counts = bigram_vect.fit_transform(train_DataFrame['text'])

#get words
unigram_features = unigram_vect.get_feature_names()
bigram_features = bigram_vect.get_feature_names()

#get the index by word
unigram_vect.vocabulary_.get('document')

tfidf_transformer = TfidfTransformer()
unigrams_tfidf = tfidf_transformer.fit_transform(unigram_counts)
unigrams_tfidf.shape