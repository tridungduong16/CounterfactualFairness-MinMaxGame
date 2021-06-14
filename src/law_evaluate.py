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
import argparse
import sys
import pprint
import gc

from utils.evaluate_func import evaluate_pred, evaluate_distribution, evaluate_fairness
from utils.helpers import load_config



def evaluate_law(df, df_result, col):
    sensitive_att = ['race', 'sex']
    target = 'ZFYA'
    for m in col:
        print(m, sensitive_att)
        performance_reg = evaluate_pred(df[m].values, df[target].values)
        performance_fairness = evaluate_fairness(sensitive_att, df, m)
        performance_reg.update(performance_fairness)
        performance_reg['method'] = m
        df_result = df_result.append(performance_reg, ignore_index=True)
    return df_result

if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_weight', type=str, default="0.1 0.5 1 1.5 2 2.5 3 3.5 4 4.5")
    parser.add_argument('--run_lambda', action='store_true')

    args = parser.parse_args()
    run_lambda = args.run_lambda
    lambda_weight = args.lambda_weight
    lambda_weight = [float(x) for x in lambda_weight.split(' ')]
    lambda_weight = [str(x) for x in lambda_weight]


    if run_lambda:
        print("Run lambda with lambda ", lambda_weight)
    else:
        print("Run normal flow")

    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)
        
    """Set up logging"""
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=conf['evaluate_law_log'])
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)
    

    """Load data"""
    # col_baseline = ["full_linear",
    #                 "full_net",
    #                 "unaware_linear",
    #                 "unaware_net",
    #                 "level2_lin_True",
    #                 "level2_lin_False",
    #                 "level3_lin_True",
    #                 "level3_lin_False"]

    # col_ivr = ['AL_prediction', 'GL_prediction', 'GD_prediction']

    # col = col_baseline +  col_ivr

    result_path = ''
    if run_lambda:
        GD_prediction = 'GD_prediction'
        col = ['GD_prediction_' + str(l)  for l in lambda_weight]
        df = pd.read_csv(conf["ivr_law_lambda"])
        result_path = conf['ivr_law_lambda']
    else:
        col = ["full_linear", "full_net",
                        "unaware_linear", "unaware_net",
                        "level2_lin_True", "level2_lin_False",
                        "level3_lin_True", "level3_lin_False",
                        "AL_prediction", "GL_prediction", "GD_prediction"]
        df2 = pd.read_csv(conf["ivr_law"])
        df1 = pd.read_csv(conf['law_baseline'])
        df2 = df2.drop(columns = ['LSAT','UGPA','ZFYA', 'race','sex'])
        df = pd.concat([df1, df2], axis=1)
        result_path = conf['evaluate_law']


    df_result = pd.DataFrame()
    df_result['method'] = ''
    df_result['RMSE'] = ''
    df_result['MAE'] = ''
    df_result['sinkhorn'] = ''
    df_result['energy'] = ''
    df_result['gaussian'] = ''
    df_result['laplacian'] = ''


    """Evaluate performance"""
    df_result = evaluate_law(df, df_result, col)
    df_result['RMSE'] = df_result['RMSE'].round(decimals=4)
    df_result['MAE'] = df_result['MAE'].round(decimals=4)
    df_result['sinkhorn'] = df_result['sinkhorn'].round(decimals=4)
    df_result['energy'] = df_result['energy'].round(decimals=4)
    df_result['gaussian'] = df_result['gaussian'].round(decimals=4)
    df_result['laplacian'] = df_result['laplacian'].round(decimals=4)
    df_result.to_csv(result_path, index = False)

    print(df_result)
    print('Collecting...')
    n = gc.collect()
    print('Unreachable objects:', n)
    print('Remaining Garbage:',)
    pprint.pprint(gc.garbage)
    # logger.debug(df_result[['method', 'RMSE', 'R2score', 'sinkhorn']])
    sys.modules[__name__].__dict__.clear()

    
    