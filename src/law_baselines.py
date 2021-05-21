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
from torch import nn
from pyro.nn import PyroModule
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
from model_arch.discriminator import Discriminator_Law


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
        posterior = pyro.infer.Importance(conditioned, num_samples=2).run()
        post_marginal = pyro.infer.EmpiricalMarginal(posterior, "Knowledge")
        post_samples = [post_marginal().item() for _ in range(2)]
        post_unique, post_counts = np.unique(post_samples, return_counts=True)
        mean = np.mean(post_samples)
        knowledge.append(mean)
    return knowledge


def infer_knowledge_counterfactual(df, feature):
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
                                                                 "LSAT": df["LSAT"][i],
                                                                 "Race": df[feature][i]})

        posterior = pyro.infer.Importance(conditioned, num_samples=2).run()
        post_marginal = pyro.infer.EmpiricalMarginal(posterior, "Knowledge")
        post_samples = [post_marginal().item() for _ in range(2)]
        post_unique, post_counts = np.unique(post_samples, return_counts=True)
        mean = np.mean(post_samples)
        knowledge.append(mean)
    return knowledge
    

def counterfactual_predict(df_cf, model, full_feature):
    y_pred = model.predict(df_cf[full_feature].values)
    return y_pred.reshape(-1)

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
    

    """Load data"""
    data_path = conf['data_law']
    df = pd.read_csv(data_path)

    """Setup features"""
    sensitive_feature = ['race', 'sex']
    normal_feature = ['LSAT', 'UGPA']
    categorical_feature = ['race', 'sex']
    full_feature = sensitive_feature + normal_feature
    target = 'ZFYA'
    selected_race = ['White', 'Black']
    df = df[df['race'].isin(selected_race)]
    df = df.reset_index(drop = True)

    """Preprocess data"""
    df['LSAT'] = (df['LSAT']-df['LSAT'].mean())/df['LSAT'].std()
    df['UGPA'] = (df['UGPA']-df['UGPA'].mean())/df['UGPA'].std()
    df['ZFYA'] = (df['ZFYA']-df['ZFYA'].mean())/df['ZFYA'].std()

    le = preprocessing.LabelEncoder()
    df['race'] = le.fit_transform(df['race'])
    df['sex'] = le.fit_transform(df['sex'])

    df['race']  = df['race'] .astype(float)
    df['sex'] = df['sex'].astype(float)

    """Full model"""
    logger.debug('Full model')

    """Counterfactual dataframe"""
    df_race0 = df.copy()
    df_race0.loc[:, 'race'] = 0

    df_race1 = df.copy()
    df_race1.loc[:, 'race'] = 1

    df_sex0 = df.copy()
    df_sex0.loc[:, 'sex'] = 0

    df_sex1 = df.copy()
    df_sex1.loc[:, 'sex'] = 1

    """Full model"""
    reg = LinearRegression().fit(df[full_feature], df['ZFYA'])
    y_pred = reg.predict(df[full_feature].values)
    df['full_prediction'] = y_pred.reshape(-1)
    df['full_prediction_race0'] = counterfactual_predict(df_race0, reg, full_feature)
    df['full_prediction_race1'] = counterfactual_predict(df_race1, reg, full_feature)
    df['full_prediction_sex0'] = counterfactual_predict(df_sex0, reg, full_feature)
    df['full_prediction_sex1'] = counterfactual_predict(df_sex1, reg, full_feature)




    """Unaware model"""
    logger.debug('Unware model')
    reg = LinearRegression().fit(df[normal_feature], df['ZFYA'])
    y_pred = reg.predict(df[normal_feature].values)
    df['unaware_prediction'] = y_pred.reshape(-1)
    df['unaware_prediction_race0'] = counterfactual_predict(df_race0, reg, normal_feature)
    df['unaware_prediction_race1'] = counterfactual_predict(df_race1, reg, normal_feature)
    df['unaware_prediction_sex0'] = counterfactual_predict(df_sex0, reg, normal_feature)
    df['unaware_prediction_sex1'] = counterfactual_predict(df_sex1, reg, normal_feature)

    """Counterfactual fairness model"""
    for i in ['LSAT', 'UGPA', 'ZFYA', 'race', 'sex']:
        df[i] = [torch.tensor(x) for x in df[i].values]

    df_race0 = df.copy()
    df_race0.loc[:, 'race'] = 0
    df_race0['race'] = df_race0['race'].astype(float)
    df_race0['race']  = [torch.tensor(x) for x in df_race0['race'].values]


    df_race1 = df.copy()
    df_race1.loc[:, 'race'] = 1
    df_race1['race'] = df_race1['race'].astype(float)
    df_race1['race'] = [torch.tensor(x) for x in df_race1['race'].values]

    df_sex0 = df.copy()
    df_sex0.loc[:, 'sex'] = 0
    df_sex0['sex'] = df_sex0['sex'].astype(float)
    df_sex0['sex'] = [torch.tensor(x) for x in df_sex0['sex'].values]

    df_sex1 = df.copy()
    df_sex1.loc[:, 'sex'] = 1
    df_sex1['sex'] = df_sex1['sex'].astype(float)
    df_sex1['sex'] = [torch.tensor(x) for x in df_sex1['sex'].values]

    logger.debug('Counterfactual fairness model')
    knowledged = infer_knowledge(df)
    knowledged = np.array(knowledged).reshape(-1,1)
    reg = LinearRegression().fit(knowledged, df['ZFYA'])

    # knowledged_race0 = np.array(infer_knowledge_counterfactual(df_race0, "race")).reshape(-1,1)
    # knowledged_race1 = np.array(infer_knowledge_counterfactual(df_race1, "race")).reshape(-1,1)
    # knowledged_sex0 = np.array(infer_knowledge_counterfactual(df_sex0, "sex")).reshape(-1,1)
    # knowledged_sex1 = np.array(infer_knowledge_counterfactual(df_sex1, "sex")).reshape(-1,1)
    #
    # y_pred = reg.predict(knowledged_race0)
    # df['cf_prediction_race0'] = y_pred.reshape(-1)
    #
    # y_pred = reg.predict(knowledged_race1)
    # df['cf_prediction_race1'] = y_pred.reshape(-1)
    #
    # y_pred = reg.predict(knowledged_sex0)
    # df['cf_prediction_sex0'] = y_pred.reshape(-1)
    #
    # y_pred = reg.predict(knowledged_sex1)
    # df['cf_prediction_sex1'] = y_pred.reshape(-1)


    df['ZFYA'] = [x.detach().numpy() for x in df['ZFYA']]
    df['LSAT'] = [x.detach().numpy() for x in df['LSAT']]
    df['UGPA'] = [x.detach().numpy() for x in df['UGPA']]

    """Output the result"""
    df.to_csv(conf['result_law_baseline'], index = False)

    sys.modules[__name__].__dict__.clear()