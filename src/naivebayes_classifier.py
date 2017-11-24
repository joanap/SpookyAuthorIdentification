# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:26:04 2017

@author: Joana Pinto
"""

from sklearn.naive_bayes import MultinomialNB
from generic_classifier import GenericClassifier 
import pandas as pd
import numpy as np

class NaiveBayesClassifier(GenericClassifier):
    
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super(NaiveBayesClassifier, self).__init__()
        self.classifier = MultinomialNB(alpha, fit_prior, class_prior)
 
    def set_classes(self, y_train):
        self.list_of_classes = np.sort(y_train.unique())
    
    def train_and_set_classes(self, X_train, y_train):
        self.train(X_train, y_train)
        self.set_classes(y_train)
    
    def predict(self, X_test):   
        return pd.DataFrame(self.classifier.predict_proba(X_test),
                     columns=self.list_of_classes)

#%%Naive Bayes Classifier as a script  
#to load data uncomment following line
#runfile('load_DataFrames.py')

if __name__ == '__main__':
    from submission_builder.write_submission import SubmissionWriter
    
    classifier = NaiveBayesClassifier()
    classifier.train_and_set_classes(X_train, y_train)
    predictions = classifier.predict(X_test)
    submission_writer = SubmissionWriter()
    submission_writer.write_submission_to_csv('../output/' + 'NaiveBayesCenas.csv',test_DataFrame.id, predictions, classifier.list_of_classes)

    


