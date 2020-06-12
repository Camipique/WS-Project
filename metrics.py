# -*- coding: utf-8 -*-
"""
@author: Carlos Quendera 49946
@author: David Pais 50220
@author: Rebekka Gorge N º 59055
            
Metrics: 
    - AUC
    - Log Loss
    - RMSE
"""

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error

from math import sqrt

def compute_auc(true_y, pred_y):
    return round(roc_auc_score(true_y, pred_y), 4)

def compute_log_loss(true_y, pred_y):
    return round(log_loss(true_y, pred_y), 4)

def compute_rmse(true_y, pred_y):
    return round(sqrt(mean_squared_error(true_y, pred_y)), 4)
