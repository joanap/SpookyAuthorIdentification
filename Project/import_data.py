# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 23:13:39 2017

@author: ASSG
"""

#%% import libraries
import numpy as np
import pandas as pd
import nltk

#%% load data
train_DataFrame = pd.read_csv('../input/train.csv')
train_DataFrame.describe()

#%% test with one example record
example_record = train_DataFrame.loc[0]

example_text = example_record.text

vector_of_words = nltk.word_tokenize(example_text)
number_of_words = len(vector_of_words)

