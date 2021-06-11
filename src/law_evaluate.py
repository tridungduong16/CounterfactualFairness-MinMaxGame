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
from utils.evaluate_func import evaluate_pred, evaluate_distribution, evaluate_fairness
from utils.helpers import load_config



def evaluate_law(df, df_result, col):
    sensitive_att = ['race']
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
    col_baseline = ["full_linear",
                    "full_net",
                    "unaware_linear",
                    "unaware_net",
                    "level2_lin_True",
                    "level2_lin_False",
                    "level3_lin_True",
                    "level3_lin_False"]

    col_ivr = ['AL_prediction', 'GL_prediction', 'GD_prediction']

    col = col_baseline +  col_ivr



    df1 = pd.read_csv(conf['result_law_baseline'])
    df2 = pd.read_csv(conf['result_ivr_law']).drop(columns = ['LSAT',
                                                              'UGPA',
                                                              'ZFYA',
                                                              'race',
                                                              'sex'])

    # print(df1.columns)

    # print(df2.columns)
    df = pd.concat([df1, df2], axis=1)


    # if mode == "baseline":
    #     df = pd.read_csv(conf['result_law_baseline'])
    #     df = df[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']]
    #     col = ['full_prediction', 'unaware_prediction', 'cf_prediction']
    # elif mode == "ivr":
    #     df = pd.read_csv(conf['result_ivr_law'])
    #     col = ['AL_prediction', 'GL_prediction', 'GD_prediction']
    # elif mode == "both":
    #     col = ['full_prediction', 'unaware_prediction', 'cf_prediction',
    #            'AL_prediction', 'GL_prediction', 'GD_prediction']
    #     df = pd.read_csv(conf['result_law_baseline'])
    #     df1 = df[['race', 'sex', 'LSAT', 'UGPA',
    #               'full_prediction', 'unaware_prediction', 'cf_prediction', 'ZFYA']]
    #     df = pd.read_csv(conf['result_ivr_law'])
    #     df2 = df[['AL_prediction', 'GL_prediction', 'GD_prediction']]
    #     df = pd.concat([df1, df2], axis=1)



    
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
    df_result.to_csv(conf['result_evaluate_law'], index = False)


    logger.debug(df_result[['method', 'RMSE', 'R2score', 'sinkhorn']])
    sys.modules[__name__].__dict__.clear()

    
    