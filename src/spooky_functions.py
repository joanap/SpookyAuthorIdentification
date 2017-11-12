# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 22:09:43 2017

@author: ASSG
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd

#%% Generic function to make predictions and output them to a valid csv file (só pq é pra ti Joana :P)

def train_predict_and_export(X_train, X_test, Y_train, id, filename):
    clf = MultinomialNB().fit(X_train, Y_train)
    prediction = clf.predict_proba(X_test)
    
    classes = Y_train.unique()
    classes.sort()
    
    submission_DF = pd.concat([id,
                               pd.DataFrame(prediction, columns = classes)], 
                        axis = 1)
    
    submission_DF.to_csv('../output/' + filename, index = False)
