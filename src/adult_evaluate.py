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
from utils.helpers import load_config
import argparse



def evaluate_adult(df, df_result, sensitive_features, label):
    for m in ['full_prediction', 'unaware_prediction','AL_prediction', 'GL_prediction', 'GD_prediction']:
        print(m)
        print(df[m].values)
        performance = {}
        performance['method'] = m
        performance_reg = evaluate_classifier(df[m].values, df[target].values)
        performance_fairness = evaluate_fairness(sensitive_features,
                                                 df,
                                                 m,
                                                 label,
                                                 problem="classification")
        performance.update(performance_reg)
        performance.update(performance_fairness)
        df_result = df_result.append(performance, ignore_index=True)


    return df_result


if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='both')

    args = parser.parse_args()
    mode = args.mode

    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

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
    if mode == "baseline":
        df = pd.read_csv(conf['result_adult'])
        df = df[['gender', 'race', 'income']]
        col = ['full_prediction', 'unaware_prediction', 'cf_prediction']
    elif mode == "ivr":
        df = pd.read_csv(conf['result_ivr_adult'])
        col = ['AL_prediction', 'GL_prediction', 'GD_prediction']
    elif mode == "both":
        col = ['full_prediction', 'unaware_prediction',
               'AL_prediction', 'GL_prediction', 'GD_prediction']
        df = pd.read_csv(conf['result_adult'])
        df1 = df[['race', 'gender', 'full_prediction', 'unaware_prediction', 'income']]
        df = pd.read_csv(conf['result_ivr_adult'])
        df2 = df[['AL_prediction', 'GL_prediction', 'GD_prediction']]
        df = pd.concat([df1, df2], axis=1)
    

    """Load data"""
    # data_path = conf['result_adult']
    # df = pd.read_csv(data_path)
    
    sensitive_features = ['race', 'gender']
    target = 'income'

    logger.debug(df)

    
    df_result = pd.DataFrame()
    df_result['method'] = ''
    df_result['F1 Score'] = ''
    df_result['Precision'] = ''
    df_result['Recall'] = ''
    df_result['Accuracy'] = ''
    df_result['dpr_race'] = ''
    df_result['eod_race'] = ''
    df_result['dpr_gender'] = ''
    df_result['eod_gender'] = ''


    """Evaluate performance"""
    df_result = evaluate_adult(df, df_result, sensitive_features, target)
    df_result.to_csv(conf['result_evaluate_adult'], index = False)
    
    logger.debug(df_result)
    sys.modules[__name__].__dict__.clear()

    
    