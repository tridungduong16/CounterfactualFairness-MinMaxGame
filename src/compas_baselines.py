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
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from torch import nn
from pyro.nn import PyroModule
from tqdm import tqdm
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting

# from utils.helpers import load_adult_income_dataset
# from utils.dataloader import DataLoader

def GroundTruthModel():
    count_dict = {'marital_status': 5,
                  'occupation': 6,
                  'race': 2,
                  'gender': 2,
                  'workclass': 4,
                  'education': 8}

    prob_education = torch.tensor([1 / count_dict['education']] * count_dict['education'])
    prob_occupation = torch.tensor([1 / count_dict['occupation']] * count_dict['occupation'])
    prob_maritalstatus = torch.tensor([1 / count_dict['marital_status']] * count_dict['marital_status'])

    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Neducation': dist.Categorical(probs=prob_education),
        'Nmarital_status': dist.Categorical(probs=prob_maritalstatus),
        'Noccupation': dist.Categorical(probs=prob_occupation),
        'Nknowledge': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nage': dist.Normal(torch.tensor(0.), torch.tensor(1.))
    }

    R = pyro.sample("Race", exo_dist['Nrace'])
    S = pyro.sample("Sex", exo_dist['Nsex'])
    E = pyro.sample("Education", exo_dist['Neducation'])
    M = pyro.sample("Marital", exo_dist['Nmarital_status'])
    O = pyro.sample("Occupation", exo_dist['Noccupation'])
    K = pyro.sample("Knowledge", exo_dist['Nknowledge'])
    A = pyro.sample("Age", exo_dist['Nage'])

    H = pyro.sample("Hour", dist.Normal(R + S + E + M + K + A, 1))
    # I = pyro.sample("Income", dist.Normal(R + S + K + E + M + H + A, 50000))


def infer_knowledge(df):
    """

    :param df: DESCRIPTION
    :type df: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """

    knowledge = []
    # df = df.sample(frac=0.1, replace=True, random_state=1).reset_index(drop=True)

    for i in tqdm(range(len(df))):
        conditioned = pyro.condition(GroundTruthModel, data={"H": df["hours_per_week"][i],
                                                             "A": df["age"][i],
                                                             "R": df["race"][i],
                                                             "S": df["gender"][i],
                                                             "O": df["occupation"][i],
                                                             "M": df["marital_status"][i]
                                                             }
                                     )
        posterior = pyro.infer.Importance(conditioned, num_samples=10).run()
        post_marginal = pyro.infer.EmpiricalMarginal(posterior, "Knowledge")
        post_samples = [post_marginal().item() for _ in range(10)]
        post_unique, post_counts = np.unique(post_samples, return_counts=True)
        mean = np.mean(post_samples)
        knowledge.append(mean)
    return knowledge


if __name__ == "__main__":
    """Device"""
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

    """Setup for dataset"""
    data_name = "compas"
    log_path = conf['log_train_compas']
    data_path = conf['data_compas']

    """Set up logging"""
    logger = setup_logging(log_path)

    """Load data"""
    df = pd.read_csv(data_path)

    """Setup features"""
    dict_ = features_setting(data_name)
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]

    """Preprocess data"""
    for c in continuous_features:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    count_dict = {}

    for c in categorical_features:
        count_dict[c] = len(df[c].unique())
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c])


    """Full model"""
    logger.debug('Full model')

    clf = LogisticRegression()
    clf.fit(df[full_features], df[target])
    y_pred = clf.predict(df[full_features].values)
    df['full_prediction'] = y_pred.reshape(-1)
    y_pred = clf.predict_proba(df[full_features].values)[:, 0]
    df['full_prediction_proba'] = y_pred.reshape(-1)

    """Unaware model"""
    logger.debug('Unware model')

    clf = LogisticRegression()
    clf.fit(df[normal_features], df[target])
    y_pred = clf.predict(df[normal_features].values)
    df['unaware_prediction'] = y_pred.reshape(-1)

    y_pred = clf.predict_proba(df[normal_features].values)[:, 0]
    df['unaware_prediction_proba'] = y_pred.reshape(-1)

    """Counterfactual fairness model"""
    df.to_csv(conf['result_compas'], index=False)



