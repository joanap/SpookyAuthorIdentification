# Clean and export data into a Data Warehouse (in csv and xlsx)

#%% import libraries
import numpy as np
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt

#%% load data
train_DataFrame = pd.read_csv('../input/train.csv')
train_DataFrame.describe()
stopwords = nltk.corpus.stopwords.words('english')

#%%
## This function receives a vector of words and returns possible words to split it
## in order to get separated sentences. (current metho: look for stopwords starting with a capital letter)
def find_split_words(vector_of_words):
    split_words = set()
    for word in vector_of_words:
        first_letter = word[0]
        if word.lower() in stopwords and first_letter == first_letter.upper():
            split_words.add(word) # if the first word's a stopword and starts 
                                     # with a capital letter probably it is 
                                     # the beggining of a new sentence
    if "I" in split_words:
        split_words.remove('I') # remove "I" word
    return split_words

#  This function receives a vector of words that represent more than one 
# sentence to be splitted and returns a vector of the splitted sentences  
def split_vector_of_words_into_sentences(vector_of_words):
    split_words = find_split_words(vector_of_words)
    number_of_words = len(vector_of_words)
    vector_of_sentences = []
    sentence_beggining = 0
    sentence_end = 1
    while sentence_end < number_of_words:
        if vector_of_words[ sentence_end ] in split_words or sentence_end == number_of_words-1:
            vector_of_sentences.append(vector_of_words[sentence_beggining : sentence_end]+['.'])
            sentence_beggining = sentence_end
            sentence_end += 1
        sentence_end += 1
    return vector_of_sentences

## create a new dataframe in which the multiple sentences have been splitted into several rows
dataframe_array = []

for index, row in train_DataFrame.iterrows():
    vector_of_sentences = split_vector_of_words_into_sentences(nltk.word_tokenize(row.text))
    for sentence in vector_of_sentences:
        dataframe_array.append([row.id,sentence,row.author])

new_train_DataFrame = pd.DataFrame(dataframe_array, columns = ['sentence_group_id','word_vector','author'])  
#save as xlsx
new_train_DataFrame.to_excel('../input/parsed_sentences.xlsx')

#join word vector into a single string and save to a csv
new_train_DataFrame_joined = new_train_DataFrame
new_train_DataFrame_joined['text'] = new_train_DataFrame_joined.word_vector.apply(lambda x: ' '.join(x))
del new_train_DataFrame_joined['word_vector']

new_train_DataFrame_joined.to_csv('../input/parsed_sentences.csv')
