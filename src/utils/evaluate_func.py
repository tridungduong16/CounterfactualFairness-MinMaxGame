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
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from collections import Counter

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
    Evaluate performance of classifier in terms of precision, recall, f-measure, accuracy
    :param y_pred: DESCRIPTION
    :type y_pred: TYPE
    :param y_true: DESCRIPTION
    :type y_true: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    print(y_true)
    print(y_pred)
    print(Counter(y_pred))
    print(Counter(y_true))

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

def fair_metrics(df, target, label, sensitive, data_name = None):
    if data_name == 'compas':
        privileged_classes = 0
    elif data_name == 'adult':
        privileged_classes = 1
    elif data_name == 'bank':
        privileged_classes = 2


    dataset = StandardDataset(df,
                              label_name=label,
                              favorable_classes=[1],
                              protected_attribute_names=[sensitive],
                              privileged_classes=[[privileged_classes]])

    dataset_pred = dataset.copy()
    dataset_pred.labels = df[target].values

    attr = dataset_pred.protected_attribute_names[0]

    idx = dataset_pred.protected_attribute_names.index(attr)
    privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
    unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

    classified_metric = ClassificationMetric(dataset,
                                             dataset_pred,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    result = {
              'true_positive_rate_difference_{}'.format(sensitive):
                  classified_metric.true_positive_rate_difference(),
              'generalized_entropy_index_{}'.format(sensitive):
                  classified_metric.generalized_entropy_index(alpha=2),
              'equal_opportunity_difference_{}'.format(sensitive):
                  classified_metric.equal_opportunity_difference(),
              'average_abs_odds_difference_{}'.format(sensitive):
                  classified_metric.average_abs_odds_difference()
              }

    return result

def evaluate_fairness(sensitive_att, df, target, label=None, problem = "regression", dataname=None):
    """
    Evaluate the fairness aspects in terms of classification and regression
    For regerssion: use the distribution distance
    For classification: use the distribution distance + EOD

    :param sensitive_att:
    :param df:
    :param target:
    :param label:
    :param problem:
    :return: dictionary of evaluation metric
    """
    eval_performance = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sinkhorn, energy, gaussian, laplacian = 0, 0, 0, 0


    # df_term = df.sample(frac=0.1, replace=True, random_state=0)
    df_term = df.copy()
    if problem == "classification":
        for s in sensitive_att:
            # ys = df_term[df_term[s] == 1][target + "_proba"].values
            # ys_hat = df_term[df_term[s] == 0][target + "_proba"].values
            # ys = torch.Tensor(ys).to(device).reshape(-1,1)
            # ys_hat = torch.Tensor(ys_hat).to(device).reshape(-1,1)
            # eval_fair = evaluate_distribution(ys, ys_hat)
            # sinkhorn += eval_fair['sinkhorn']
            # energy += eval_fair['energy']
            # gaussian += eval_fair['gaussian']
            # laplacian += eval_fair['laplacian']
            eval = fair_metrics(df, target, label, s, dataname)
            eval_performance.update(eval)

    elif problem == "regression":
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
            # del ys
            # del ys_hat

        eval_performance['sinkhorn'] = sinkhorn
        eval_performance['energy'] = energy
        eval_performance['gaussian'] = gaussian
        eval_performance['laplacian'] = laplacian
    return eval_performance

def evaluate_classification_performance(df,
                                        df_result,
                                        sensitive_features,
                                        label,
                                        prediction_columns,
                                        data_name):
    for m in prediction_columns:
        performance = {}
        performance['method'] = m
        print(m, label)
        performance_reg = evaluate_classifier(df[m].values, df[label].values)
        performance_fairness = evaluate_fairness(sensitive_features,
                                                 df,
                                                 target = m,
                                                 label = label,
                                                 problem="classification",
                                                 dataname=data_name)
        performance.update(performance_reg)
        performance.update(performance_fairness)
        df_result = df_result.append(performance, ignore_index=True)
    return df_result