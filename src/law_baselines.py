#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:55:24 2021
@author: trduong
"""

import pandas as pd
import numpy as np
import logging
import yaml
import pyro
import torch 
import pyro.distributions as dist
import sys

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import preprocess_dataset
from utils.helpers import features_setting

def GroundTruthModel():
    """
    

    Returns
    -------
    None.

    """
    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Nknowledge': dist.Normal(torch.tensor(0.), torch.tensor(1.))
    }
    
    R = pyro.sample("Race", exo_dist['Nrace'])
    S = pyro.sample("Sex", exo_dist['Nsex'])
    K = pyro.sample("Knowledge", exo_dist['Nknowledge'])
    
    # PsuedoDelta 
    G = pyro.sample("UGPA", dist.Normal(K + R + S, 0.1))
    L = pyro.sample("LSAT", dist.Normal(K + R + S, 0.1))
    F = pyro.sample("ZFYA", dist.Normal(K + R + S, 0.1))


def infer_knowledge(df):
    """
    
    :param df: DESCRIPTION
    :type df: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """
    
    knowledge = []
    # df = df.sample(frac=0.5, replace=True, random_state=1).reset_index(drop=True)

    for i in tqdm(range(len(df))):
        conditioned = pyro.condition(GroundTruthModel, data={"UGPA": df["UGPA"][i],
                                                             "LSAT": df["LSAT"][i]})
        posterior = pyro.infer.Importance(conditioned, num_samples=10).run()
        post_marginal = pyro.infer.EmpiricalMarginal(posterior, "Knowledge")
        post_samples = [post_marginal().item() for _ in range(10)]
        post_unique, post_counts = np.unique(post_samples, return_counts=True)
        mean = np.mean(post_samples)
        knowledge.append(mean)
    return knowledge


if __name__ == "__main__":
    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

    """Set up logging"""
    logger = setup_logging(conf['log_baselines_law'])

    """Load data"""
    data_path = conf['data_law']
    df = pd.read_csv(data_path)
    """Setup features"""
    data_name = "law"
    dict_ = features_setting("law")
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]
    selected_race = ['White', 'Black']
    df = df[df['race'].isin(selected_race)]
    df = df.reset_index(drop = True)

    """Preprocess data"""
    df = preprocess_dataset(df, continuous_features, categorical_features)
    df['ZFYA'] = (df['ZFYA']-df['ZFYA'].mean())/df['ZFYA'].std()


    le = preprocessing.LabelEncoder()
    df['race'] = le.fit_transform(df['race'])
    df['sex'] = le.fit_transform(df['sex'])
    df['race']  = df['race'] .astype(float)
    df['sex'] = df['sex'].astype(float)

    # df, df_test = train_test_split(df, test_size=0.1, random_state=0)
    # df = df.reset_index(drop = True)
    # df_test = df_test.reset_index(drop = True)
    # df_test = df_test[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']]
    df, df_test = train_test_split(df, test_size=0.1, random_state=0)
    print(df[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']])
    sys.exit(1)

    # df_test = df.copy()
    """Full model"""
    logger.debug('Full model')

    """Full model"""
    reg = LinearRegression().fit(df[full_feature], df['ZFYA'])
    y_pred = reg.predict(df_test[full_feature].values)
    df_test['full_prediction'] = y_pred.reshape(-1)

    """Unaware model"""
    logger.debug('Unware model')
    reg = LinearRegression().fit(df[normal_feature], df['ZFYA'])
    y_pred = reg.predict(df_test[normal_feature].values)
    df_test['unaware_prediction'] = y_pred.reshape(-1)

    """Counterfactual fairness model"""
    for i in ['LSAT', 'UGPA', 'race', 'sex']:
        df[i] = [torch.tensor(x) for x in df[i].values]
        df_test[i] = [torch.tensor(x) for x in df_test[i].values]

    logger.debug('Counterfactual fairness model')
    knowledged = infer_knowledge(df)
    knowledged = np.array(knowledged).reshape(-1,1)
    reg = LinearRegression().fit(knowledged, df['ZFYA'])

    knowledged = infer_knowledge(df_test)
    knowledged = np.array(knowledged).reshape(-1,1)
    df_test['cf_prediction'] = reg.predict(knowledged)

    # df_test['ZFYA'] = [x.detach().numpy() for x in df_test['ZFYA']]
    df_test['LSAT'] = [x.detach().numpy() for x in df_test['LSAT']]
    df_test['UGPA'] = [x.detach().numpy() for x in df_test['UGPA']]
    df_test['race'] = [x.detach().numpy() for x in df_test['race']]
    df_test['sex'] = [x.detach().numpy() for x in df_test['sex']]

    """Output the result"""
    df_test.to_csv(conf['result_law_baseline'], index = False)
    sys.modules[__name__].__dict__.clear()