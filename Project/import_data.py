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

#%% Create len feature for the whole dataset
stopwords = nltk.corpus.stopwords.words('english')

train_DataFrame['word_vector'] = train_DataFrame.text.apply(nltk.word_tokenize)
train_DataFrame['word_count'] = train_DataFrame.word_vector.apply(len)


#%% analyse new features
authors = train_DataFrame.author.unique()
mean_word_count = [train_DataFrame[train_DataFrame.author == auth ].word_count.mean() for auth in authors]

plt.bar(range(len(authors)), mean_word_count)
plt.xticks(range(len(authors)), authors)
