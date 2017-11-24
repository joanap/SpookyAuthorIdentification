# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:18:02 2017

@author: Joana Pinto
"""

from abc import ABCMeta
from abc import abstractmethod

class GenericClassifier:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
    
    @abstractmethod
    def predict(self, X_test):
        return
        
    def build_submission(self, id, predict):
        submission_DF = pd.concat([id,
                                   pd.DataFrame(prediction, columns = classes)], 
                       axis = 1)
    
    submission_DF.to_csv('../output/' + filename, index = False)