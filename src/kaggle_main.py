# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:49:17 2017

@author: Joana Pinto
"""
import pandas as pd
from submission_builder.write_submission import SubmissionWriter
from features.bag_of_words import BagOfWords
from features.word_balancer import WordBalancer
from naivebayes_classifier import NaiveBayesClassifier

def main():


    #%% import data
    train_DataFrame = pd.read_excel('../input/parsed_sentences.xlsx')
    test_DataFrame = pd.read_csv('../input/test.csv')
    y_train = train_DataFrame.author
    
    #%% bag of words
    baf_of_words = BagOfWords(1,2,r'\b\w\w+\b')
    baf_of_words.prepare(train_DataFrame.text)
    train_transform_result_bigram  = baf_of_words.cook(train_DataFrame.text)
    test_transform_result_bigram  = baf_of_words.cook(test_DataFrame.text)
    
    #%% tf_idf
    tf_idf = WordBalancer()
    tf_idf.prepare(train_transform_result_bigram)
    train_tf_idf = tf_idf.cook(train_transform_result_bigram)
    test_tf_idf = tf_idf.cook(test_transform_result_bigram)
    
    #%% call classifier
    classifier = NaiveBayesClassifier()
    classifier.train_and_set_classes(train_tf_idf, y_train)
    predictions = classifier.predict(test_tf_idf)
    
    #%%create submission file
    submission_writer = SubmissionWriter()
    submission_writer.write_submission_to_csv('../output/' + 'NaiveBayesCenas.csv',test_DataFrame.id, predictions, classifier.list_of_classes)
    

if __name__ == "__main__":
    main()