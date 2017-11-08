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
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# our beautiful functions:
from spooky_functions import train_predict_and_export

#%% import data
train_DataFrame = pd.read_excel('../input/parsed_sentences.xlsx')
train_DataFrame.describe()

test_DataFrame = pd.read_csv('../input/test.csv')
#%%
#default constructor
# CountVectorizer(analyzer=...'word', binary=False, decode_error=...'strict',
#        dtype=<... 'numpy.int64'>, encoding=...'utf-8', input=...'content',
#        lowercase=True, max_df=1.0, max_features=None, min_df=1,
#        ngram_range=(1, 1), preprocessor=None, stop_words=None,
#        strip_accents=None, token_pattern=...'(?u)\\b\\w\\w+\\b',
#        tokenizer=None, vocabulary=None)

unigram_vect = CountVectorizer()

Ngram_vect = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w\w+\b', 
                              min_df=2,
                              stop_words='english')


#get word occurences per sentence
unigram_counts = unigram_vect.fit_transform(train_DataFrame['text'])
Ngram_counts = Ngram_vect.fit_transform(train_DataFrame['text'])

#get words
unigram_features = unigram_vect.get_feature_names()
Ngram_features = Ngram_vect.get_feature_names()
print("Number of Ngrams found {0}".format(len(Ngram_features)))

unigram_counts_test = unigram_vect.transform(test_DataFrame['text'])
Ngram_count_test = Ngram_vect.transform(test_DataFrame['text'])
#%%
#get the index by word
unigram_vect.vocabulary_.get('document')

#%% Apply term-frequency and inverse-document-frequency fixes
tfidf_transformer = TfidfTransformer()
unigrams_tfidf = tfidf_transformer.fit_transform(unigram_counts)
unigrams_tfidf_test = tfidf_transformer.transform(unigram_counts_test)
Ngrams_tfidf = tfidf_transformer.fit_transform(Ngram_counts)
Ngrams_tfidf_test = tfidf_transformer.transform(Ngram_count_test)

#%% Train and predict using unigrams
X_train = unigrams_tfidf
X_test = unigrams_tfidf_test
y_train = train_DataFrame.author
row_id = test_DataFrame.id

train_predict_and_export(
            X_train, 
            X_test, 
            y_train, 
            row_id, 
            'submission_NaiveBayses.csv')

#%% Train and predict using N-grams
X_train = Ngrams_tfidf 
X_test = Ngrams_tfidf_test

train_predict_and_export(
        X_train, 
        X_test, 
        y_train, 
        row_id, 
        'submission_NaiveBayses_bigrams.csv')