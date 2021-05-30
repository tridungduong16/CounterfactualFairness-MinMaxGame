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

    """Load configuration"""
    with open("/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml", 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    """Set up logging"""
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=conf['log_law'])
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)

    """Load data and dataloader and normalize data"""
    # df = load_adult_income_dataset(conf['data_adult'])
    df = pd.read_csv(conf['processed_data_adult'])
    df = df.dropna()
    print(df.shape)
    # df = df.sample(frac=0.2, replace=True, random_state=1).reset_index(drop=True)

    """Setup features"""
    categorical_features = ['marital_status', 'occupation', 'race', 'gender', 'workclass', 'education']
    continuous_features = ['age', 'hours_per_week']
    normal_features = ['age', 'workclass', 'marital_status', 'occupation', 'hours_per_week']
    full_features = ['age', 'workclass', 'education', 'marital_status', 'occupation', 'hours_per_week', 'race',
                     'gender']
    sensitive_features = ['race', 'gender']
    target = 'income'

    """Preprocess data"""
    for c in continuous_features:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    count_dict = {}

    for c in categorical_features:
        count_dict[c] = len(df[c].unique())
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c])

    # peducation = torch.tensor([1/4, 1/4, 1/4, 1/4])
    # print(dist.Multinomial(total_count=100, probs=peducation))

    """Full model"""
    logger.debug('Full model')

    clf = LogisticRegression()
    # clf = lgb.LGBMClassifier()
    clf.fit(df[full_features], df[target])
    y_pred = clf.predict(df[full_features].values)
    df['full_prediction'] = y_pred.reshape(-1)

    # print(df)
    # print(clf.predict(df[full_features].values))
    y_pred = clf.predict_proba(df[full_features].values)[:, 0]
    # print(len(y_pred))
    df['full_prediction_proba'] = y_pred.reshape(-1)

    """Unaware model"""
    logger.debug('Unware model')

    clf = LogisticRegression()
    # clf = lgb.LGBMClassifier()
    clf.fit(df[normal_features], df[target])
    y_pred = clf.predict(df[normal_features].values)
    df['unaware_prediction'] = y_pred.reshape(-1)

    y_pred = clf.predict_proba(df[normal_features].values)[:, 0]
    df['unaware_prediction_proba'] = y_pred.reshape(-1)

    """Counterfactual fairness model"""
    # for i in full_features:
    #     df[i] = [torch.tensor(x) for x in df[i].values]
    #
    # logger.debug('Counterfactual fairness model')
    # knowledged = infer_knowledge(df)
    # knowledged = np.array(knowledged).reshape(-1, 1)
    # # print(knowledged)
    # clf = lgb.LGBMClassifier()
    # clf.fit(knowledged, df[target])
    # y_pred = clf.predict(knowledged)
    # df['cf_prediction'] = y_pred.reshape(-1)
    #
    # for i in full_features:
    #     df[i] = [x.detach().numpy() for x in df[i]]

    df.to_csv(conf['result_adult'], index=False)



