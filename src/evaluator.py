# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:28:15 2017

@author: Joana Pinto
"""
from sklearn.metrics import log_loss

#y_true: vector of labels
#y_pred: vector of vectors with the likelihood for each class
def logLoss(y_true,y_pred):
    return log_loss(y_true, y_pred)
