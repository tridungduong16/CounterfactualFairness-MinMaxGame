#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:17:22 2021

@author: trduong
"""

# import os, sys;
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import pandas as pd
import logging
import argparse
import sys

from utils.evaluate_func import evaluate_distribution, evaluate_fairness, evaluate_classifier, evaluate_classification_performance
from utils.helpers import load_config
from utils.helpers import features_setting
if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, default='both')
    # parser.add_argument('--data_name', type=str, default='compas')
    # parser.add_argument('--data_name', type=str, default='adult')
    # parser.add_argument('--data_name', type=str, default='bank')

    args = parser.parse_args()
    data_name = args.data_name

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

    dict_ = features_setting(data_name)
    sensitive_features = dict_['sensitive_features']
    label = dict_['target']

    baseline_path = conf['result_{}'.format(data_name)]
    ivr_path = conf['result_ivr_{}'.format(data_name)]
    evaluate_path = conf['result_evaluate_{}'.format(data_name)]

    baseline_columns = ['full_prediction', 'unaware_prediction']
    method_columns = ['AL_prediction', 'GL_prediction', 'GD_prediction']

    baseline_columns_proba = ['AL_prediction_proba', 'GL_prediction_proba', 'GD_prediction_proba']
    method_columns_proba = ['AL_prediction_proba', 'GL_prediction_proba', 'GD_prediction_proba']

    prediction_columns = baseline_columns + method_columns



    """Load data"""
    df = pd.read_csv(baseline_path)
    df1 = df[sensitive_features + ['full_prediction',
                                   'full_prediction_proba',
                                   'unaware_prediction',
                                   'unaware_prediction_proba',
                                   label]]
    df = pd.read_csv(ivr_path)
    df2 = df[['AL_prediction', 'AL_prediction_proba',
              'GL_prediction', 'GL_prediction_proba',
              'GD_prediction', 'GD_prediction_proba']]
    df = pd.concat([df1, df2], axis=1)

    df_result = pd.DataFrame()

    """Evaluate performance"""
    df_result = evaluate_classification_performance(df,
                                                    df_result,
                                                    sensitive_features,
                                                    label,
                                                    prediction_columns,
                                                    data_name)

    cols = list(df_result)
    cols.insert(0, cols.pop(cols.index('method')))
    df_result = df_result[cols]
    df_result.to_csv(evaluate_path, index = False)
    
    logger.debug(df_result)
    sys.modules[__name__].__dict__.clear()

    
    