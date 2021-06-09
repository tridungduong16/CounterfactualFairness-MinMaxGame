#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:55:24 2021
@author: trduong
"""

import pandas as pd
import numpy as np
import logging
import pyro
import torch
import pyro.distributions as dist
import sys
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting


def GroundTruthModel_Adult():
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
    K1 = pyro.sample("Knowledge1", exo_dist['Nknowledge'])
    K2 = pyro.sample("Knowledge2", exo_dist['Nknowledge'])
    K3 = pyro.sample("Knowledge3", exo_dist['Nknowledge'])
    K4 = pyro.sample("Knowledge4", exo_dist['Nknowledge'])
    A = pyro.sample("Age", exo_dist['Nage'])

    mu_E = R + S + K1 + K2 + K3 + K4
    x1, x2, x3, x4, x5, x6, x7 = mu_E - 3, mu_E - 2, mu_E - 1, mu_E, mu_E + 1, mu_E + 2, mu_E + 3
    E = pyro.sample("Education", dist.Normal(mu_E, 1))
    if (E >= x7) or (E <= x1):
        E = np.random.choice([2,5])
    elif (E>= x1 and E <= x2) or (E>= x6 and E <= x7):
        E = np.random.choice([0,6,4])
    elif (E>= x2 and E <= x3) or (E>= x5 and E <= x6):
        E = np.random.choice([1,7])
    elif (E >= x3) and (E <= x5):
        E = np.random.choice([3])
    E = torch.tensor(E)

    mu_M = R + S + E + K1 + K2 + K3 + K4
    x1, x2, x3, x4, x5, x6, x7 = mu_M - 3, mu_M - 2, mu_M - 1, mu_M, mu_M + 1, mu_M + 2, mu_M + 3
    M = pyro.sample("Marital", dist.Normal(mu_M, 1))
    if (M >= x7) or (M <= x1):
        M = 4
    elif (M>= x1 and M <= x2) or (M>= x6 and M <= x7):
        M = np.random.choice([0,2])
    elif (M>= x2 and M <= x3) or (M>= x5 and M <= x6):
        M = 3
    elif (M >= x3) and (M <= x5):
        M = 1
    M = torch.tensor(M)

    mu_O = R + S + E + K1 + K2 + K3 + K4
    x1, x2, x3, x4, x5, x6, x7 = mu_O - 3, mu_O - 2, mu_O - 1, mu_O, mu_O + 1, mu_O + 2, mu_O + 3
    O = pyro.sample("Occupation", dist.Normal(mu_O, 1))
    if (O >= x7) or (O <= x1):
        O = 1
    elif (O>= x1 and O <= x2) or (O>= x6 and O <= x7):
        O = np.random.choice([2,3])
    elif (O>= x2 and O <= x3) or (O>= x5 and O <= x6):
        O = np.random.choice([4,5])
    elif (M >= x3) and (O <= x5):
        O = 0
    O = torch.tensor(O)

    mu_H = R + S + E + M + O + K1 + K2 + K3 + K4
    H = pyro.sample("Hour", dist.Normal(mu_H, 1))


    # M = pyro.sample("Marital", dist.Normal(R + S + E + M + K + A + O, 1))
    # O = pyro.sample("Occupation", dist.Normal(R + S + E + M + K + A + O, 1))
    # H = pyro.sample("Hour", dist.Normal(R + S + E + M + K1 + K2 + K3 + K4 , 1))
    # R = pyro.sample("Race", exo_dist['Nrace'])
    # S = pyro.sample("Sex", exo_dist['Nsex'])
    # E = pyro.sample("Education", exo_dist['Neducation'])
    # M = pyro.sample("Marital", exo_dist['Nmarital_status'])
    # O = pyro.sample("Occupation", exo_dist['Noccupation'])
    # K = pyro.sample("Knowledge", exo_dist['Nknowledge'])
    # A = pyro.sample("Age", exo_dist['Nage'])
    # H = pyro.sample("Hour", dist.Normal(R + S + E + M + K + A + O, 1))

def infer_knowledge_adult(df):
    """

    :param df: DESCRIPTION
    :type df: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """

    knowledge1 = []
    knowledge2 = []
    knowledge3 = []

    for i in tqdm(range(len(df))):
        conditioned = pyro.condition(GroundTruthModel_Adult, data={"H": df["hours_per_week"][i],
                                                             "A": df["age"][i],
                                                             "R": df["race"][i],
                                                             "S": df["gender"][i],
                                                             "O": df["occupation"][i],
                                                             "M": df["marital_status"][i]
                                                             }
                                     )
        posterior = pyro.infer.Importance(conditioned, num_samples=10).run()

        post_marginal_1 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge1")
        post_marginal_2 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge2")
        post_marginal_3 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge3")

        post_samples1 = [post_marginal_1().item() for _ in range(10)]
        post_samples2 = [post_marginal_2().item() for _ in range(10)]
        post_samples3 = [post_marginal_3().item() for _ in range(10)]

        mean1 = np.mean(post_samples1)
        mean2 = np.mean(post_samples2)
        mean3 = np.mean(post_samples3)

        knowledge1.append(mean1)
        knowledge2.append(mean2)
        knowledge3.append(mean3)

    knowledge1 = np.array(knowledge1).reshape(-1, 1)
    knowledge2 = np.array(knowledge2).reshape(-1, 1)
    knowledge3 = np.array(knowledge3).reshape(-1, 1)
    knowledged = np.concatenate((knowledge1, knowledge2, knowledge3), axis=1)
    return knowledged

def GroundTruthModel_Compas():
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
    H = pyro.sample("Hour", dist.Normal(R + S + E + M + K + A + O, 1))

def infer_knowledge_compas(df):
    """

    :param df: DESCRIPTION
    :type df: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """

    knowledge = []

    for i in tqdm(range(len(df))):
        conditioned = pyro.condition(GroundTruthModel_Adult, data={"H": df["hours_per_week"][i],
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
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_name', type=str, default='compas')
    parser.add_argument('--data_name', type=str, default='adult')
    # parser.add_argument('--data_name', type=str, default='bank')
    args = parser.parse_args()

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

    dataname = args.data_name

    log_path = conf['log_train_{}'.format(dataname)]
    data_path = conf['data_{}'.format(dataname)]

    """Set up logging"""
    logger = setup_logging(log_path)

    """Load data"""
    df = pd.read_csv(data_path)

    """Setup features"""
    dict_ = features_setting(dataname)
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
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(c)
        print(le_name_mapping)
        print(df[c].value_counts())

    """Split dataset into train and test"""
    df, df_test = train_test_split(df, test_size=0.1, random_state=0)
    df = df.reset_index(drop = True)
    df_test = df_test.reset_index(drop=True)

    print(df.shape, df_test.shape)

    """Full model"""
    logger.debug('Full model')
    clf = LogisticRegression()
    clf.fit(df[full_features], df[target])
    y_pred = clf.predict(df_test[full_features].values)
    df_test['full_prediction'] = y_pred.reshape(-1)
    y_pred = clf.predict_proba(df_test[full_features].values)[:, 0]
    df_test['full_prediction_proba'] = y_pred.reshape(-1)

    """Unaware model"""
    logger.debug('Unware model')
    clf = LogisticRegression()
    clf.fit(df[normal_features], df[target])
    y_pred = clf.predict(df_test[normal_features].values)
    df_test['unaware_prediction'] = y_pred.reshape(-1)
    y_pred = clf.predict_proba(df_test[normal_features].values)[:, 0]
    df_test['unaware_prediction_proba'] = y_pred.reshape(-1)

    """Counterfactual fairness model"""
    print(df.shape)
    knowledged = infer_knowledge_adult(df)
    df_test.to_csv(conf['result_{}'.format(dataname)], index=False)


