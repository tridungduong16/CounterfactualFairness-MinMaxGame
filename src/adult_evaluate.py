#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:17:22 2021

@author: trduong
"""

# import os, sys;
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import pandas as pd
import numpy as np
import logging
import yaml
import pyro
import torch 
import pyro.distributions as dist
import sys
from utils.evaluate_func import evaluate_distribution, evaluate_fairness, evaluate_classifier




def evaluate_adult(df, df_result, sensitive_features, target):
    df = df.sample(frac=0.1, replace=True, random_state=1)
    for m in ['full_prediction', 'unaware_prediction']:
        performance_reg = evaluate_classifier(df[m].values, df[target].values)
        m_fair = m + "_proba"
        performance_fairness = evaluate_fairness(sensitive_features, df, m_fair)
        performance_reg.update(performance_fairness)
        performance_reg['method'] = m
        df_result = df_result.append(performance_reg, ignore_index=True)
    return df_result

if __name__ == "__main__":    
    """Load configuration"""
    with open("/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml", 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    """Set up logging"""
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=conf['evaluate_law'])
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)
    

    """Load data"""
    data_path = conf['result_adult']
    df = pd.read_csv(data_path)
    
    sensitive_features = ['race', 'gender']
    target = 'income'

    logger.debug(df)

    
    df_result = pd.DataFrame()
    df_result['method'] = ''
    df_result['F1 Score'] = ''
    df_result['Precision'] = ''
    df_result['Recall'] = ''
    df_result['sinkhorn'] = ''
    df_result['energy'] = ''
    df_result['gaussian'] = ''
    

    """Evaluate performance"""
    df_result = evaluate_adult(df, df_result, sensitive_features, target)
    df_result.to_csv(conf['result_evaluate_adult'], index = False)
    
    logger.debug(df_result)
    sys.modules[__name__].__dict__.clear()

    
    