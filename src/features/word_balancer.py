# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:32:29 2017

@author: Joana Pinto
"""

from sklearn.feature_extraction.text import TfidfTransformer
from features.abstract_features import AbstractFeatures 

class WordBalancer(AbstractFeatures):
    def __init__(self):
        self.balancer = TfidfTransformer()
    
    def prepare(self, input_data):
        self.balancer.fit(input_data)
    
    def cook(self, input_data):
        return self.balancer.transform(input_data)
