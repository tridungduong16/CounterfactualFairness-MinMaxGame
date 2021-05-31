#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:01:51 2021

@author: trduong
"""


import pandas as pd
import numpy as np
import logging
import yaml
import torch
import sys

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from aif360.metrics import ClassificationMetric

from tqdm import tqdm
from geomloss import SamplesLoss
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio

def evaluate_pred(y_pred, y_true):
    """
    
    :param y_pred: DESCRIPTION
    :type y_pred: TYPE
    :param y_true: DESCRIPTION
    :type y_true: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    evaluations = {}
    evaluations['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    evaluations['MAE'] = mean_absolute_error(y_true, y_pred)
    evaluations['R2score'] = r2_score(y_true, y_pred)


    return evaluations

def evaluate_classifier(y_pred, y_true):
    """
    
    :param y_pred: DESCRIPTION
    :type y_pred: TYPE
    :param y_true: DESCRIPTION
    :type y_true: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    evaluations = {}
    evaluations['F1 Score'] = f1_score(y_true, y_pred, average='weighted')
    evaluations['Precision'] = precision_score(y_true, y_pred, average='weighted')
    evaluations['Recall'] = recall_score(y_true, y_pred, average='weighted')
    evaluations['Accuracy'] = accuracy_score(y_true, y_pred)



    return evaluations

def evaluate_distribution(ys, ys_hat):
    """
    :param ys: DESCRIPTION
    :type ys: TYPE
    :param ys_hat: DESCRIPTION
    :type ys_hat: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """
    # print(ys)
    #
    # print(ys_hat)
    evaluation = {}

    backend = "auto"

    Loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.95, backend = backend)
    evaluation['sinkhorn'] = Loss(ys, ys_hat).cpu().detach().numpy() 
    
    Loss = SamplesLoss("energy", p=2, blur=0.05, scaling=0.95, backend = backend)
    evaluation["energy"] = Loss(ys, ys_hat).cpu().detach().numpy() 
    
    Loss = SamplesLoss("gaussian", p=2, blur=0.5, scaling=0.95, backend = backend)
    evaluation["gaussian"] = Loss(ys, ys_hat).cpu().detach().numpy()

    Loss = SamplesLoss("laplacian", p=2, blur=0.5, scaling=0.95, backend = backend)
    evaluation["laplacian"] = Loss(ys, ys_hat).cpu().detach().numpy()



    
    return evaluation 

def evaluate_fairness(sensitive_att, df, target, label=None, problem = "regression"):
    eval_performance = {}

    if problem == "classification":
        for s in sensitive_att:
            y_true = df[label].values
            y_pred = df[target].values
            sensitive_value = df[s].values
            eval_performance["dpr_" + s] = demographic_parity_ratio(y_true,y_pred,sensitive_features = sensitive_value)
            eval_performance["dpd_" + s] = demographic_parity_difference(y_true,y_pred,sensitive_features = sensitive_value)
            eval_performance["eod_" + s] = equalized_odds_difference(y_true,y_pred,sensitive_features = sensitive_value)
            eval_performance["eor_" + s] = equalized_odds_ratio(y_true,y_pred,sensitive_features = sensitive_value)

        return eval_performance

    if problem == "regression":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sinkhorn, energy, gaussian, laplacian = 0,0,0,0

        df = df.sample(frac=0.5, replace=True, random_state=0)

        for s in sensitive_att:
            ys = df[df[s] == 1][target].values
            ys_hat = df[df[s] == 0][target].values

            ys = torch.Tensor(ys).to(device).reshape(-1,1)
            ys_hat = torch.Tensor(ys_hat).to(device).reshape(-1,1)

            eval_performance = evaluate_distribution(ys, ys_hat)
            sinkhorn += eval_performance['sinkhorn']
            energy += eval_performance['energy']
            gaussian += eval_performance['gaussian']
            laplacian += eval_performance['laplacian']

            del ys
            del ys_hat

        eval_performance['sinkhorn'] = sinkhorn
        eval_performance['energy'] = energy
        eval_performance['gaussian'] = gaussian
        eval_performance['laplacian'] = laplacian

        return eval_performance
    
    
