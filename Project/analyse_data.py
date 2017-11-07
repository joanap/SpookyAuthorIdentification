# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:36:16 2017

@author: ASSG
"""
#%% import libraries
import numpy as np
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt

#%% Import already cleaned data
train_DF = pd.read_excel('../input/parsed_sentences.xlsx')
stopwords = nltk.corpus.stopwords.words('english')

#%% Create len feature for the whole dataset
def count_stop_words(vector_of_words):
    vector_of_stopwords = [i for i in vector_of_words if (i.lower() in stopwords)] 
    return len( vector_of_stopwords )

train_DF['word_count'] = train_DF.word_vector.apply(len)
train_DF['stopword_count'] = train_DF.word_vector.apply(count_stop_words)
train_DF['non_stopword_count'] = train_DF.word_count - train_DF.stopword_count

def histogram_of_wordcount(word):
    df = train_DF[['word_vector','author']]
    df['word_count'] = df.word_vector.apply(lambda x: x.count(word))
    print(df.groupby('author').describe())
    df.groupby('author').hist()
histogram_of_wordcount('I')    

#%% analyse new features
DF_grouped_by_author = train_DF.groupby("author")
DF_grouped_by_author.word_count.describe()
DF_grouped_by_author.stopword_count.describe()
DF_grouped_by_author.non_stopword_count.describe()

DF_grouped_by_author.hist(log = True)
##NOTE some sentences have more than 100 characters -> more preprocessing required

#%% Identify unsplitted sentences
big_sentences_DF = train_DF[train_DF.word_count > 100 ]

# are first words in sentences stop words? if so we could use this to split big sentences: if is Upper case and is stop word then split...
def find_first_word(vector_of_words):
    # function to identify the first wordin a sentence (ignores quotes and other non-letter characters)
    i = 0    
    while not re.match(r'\w.*', vector_of_words[i]):
        i +=1
    return vector_of_words[i]


number_of_sent_starting_with_stopword = train_DF.word_vector.apply(find_first_word).apply(lambda x: x.lower() in stopwords).sum()
print("{0} out of {1} sentences start with a stop word".format(
        number_of_sent_starting_with_stopword,
        train_DF.count()[0]))
