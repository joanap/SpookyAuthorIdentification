# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 23:13:39 2017

@author: ASSG
"""

#%% import libraries
import numpy as np
import pandas as pd
import nltk

import matplotlib.pyplot as plt

#%% load data
train_DataFrame = pd.read_csv('../input/train.csv')
train_DataFrame.describe()

#%% test with one example record
example_record = train_DataFrame.loc[0]

example_text = example_record.text

vector_of_words = nltk.word_tokenize(example_text)
print(example_text)
print(vector_of_words)
number_of_words = len(vector_of_words)
vector_of_stopwords = [i for i in vector_of_words if (i in stopwords)] 
number_of_stop_words = len( vector_of_stopwords )

#%% Create len feature for the whole dataset
stopwords = nltk.corpus.stopwords.words('english')

def count_stop_words(vector_of_words):
    vector_of_stopwords = [i for i in vector_of_words if (i.lower() in stopwords)] 
    return len( vector_of_stopwords )

train_DataFrame['word_vector'] = train_DataFrame.text.apply(nltk.word_tokenize)
train_DataFrame['word_count'] = train_DataFrame.word_vector.apply(len)
train_DataFrame['stopword_count'] = train_DataFrame.word_vector.apply(count_stop_words)
train_DataFrame['non_stopword_count'] = train_DataFrame.word_count - train_DataFrame.stopword_count

#%% analyse new features
DF_grouped_by_author = train_DataFrame.groupby("author")
DF_grouped_by_author.word_count.describe()
DF_grouped_by_author.stopword_count.describe()
DF_grouped_by_author.non_stopword_count.describe()

DF_grouped_by_author.hist(log = True)
##NOTE some sentences have more than 100 characters -> more preprocessing required

