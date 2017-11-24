# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:18:02 2017

@author: Joana Pinto
"""

from abc import ABCMeta
from abc import abstractmethod
import pandas as pd

class GenericClassifier:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self.list_of_classes = []
        
    @abstractmethod
    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
    
    @abstractmethod
    def predict(self, X_test):
        return
        
