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

#%% Identify unsplitted sentences
big_sentences_DF = train_DataFrame[train_DataFrame.word_count > 100 ]

# are first words in sentences stop words? if so we could use this to split big sentences: if is Upper case and is stop word then split...
def find_first_word(vector_of_words):
    # function to identify the first wordin a sentence (ignores quotes and other non-letter characters)
    i = 0    
    while not re.match(r'\w.*', vector_of_words[i]):
        i +=1
    return vector_of_words[i]

number_of_sent_starting_with_stopword = train_DataFrame.word_vector.apply(find_first_word).apply(lambda x: x.lower() in stopwords).sum()
print("{0} out of {1} sentences start with a stop word".format(
        number_of_sent_starting_with_stopword,
        train_DataFrame.count()[0]))
