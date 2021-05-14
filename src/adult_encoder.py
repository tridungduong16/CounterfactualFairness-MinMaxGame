#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:58:19 2020

@author: trduong
"""

import sys
import pandas as pd
import argparse
import torch
import yaml 
import logging 

from dfencoder.autoencoder import AutoEncoder
from sklearn import preprocessing

if __name__ == "__main__":
    """Device"""
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

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
    df = df.sample(frac=0.2, replace=True, random_state=1).reset_index(drop=True)

    """Setup features"""
    categorical_features = ['marital_status', 'occupation', 'race', 'gender', 'workclass', 'education']
    continuous_features = ['age', 'hours_per_week']
    normal_features = ['age', 'workclass', 'marital_status', 'occupation', 'hours_per_week']
    full_features = ['age', 'workclass', 'education', 'marital_status', 'occupation', 'hours_per_week', 'race',
                     'gender']
    sensitive_features = ['race', 'gender']
    target = 'income'
    
    df_generator = df[normal_feature]

    
    """Preprocess data"""
    df['LSAT'] = (df['LSAT']-df['LSAT'].mean())/df['LSAT'].std()
    df['UGPA'] = (df['UGPA']-df['UGPA'].mean())/df['UGPA'].std()
    df['ZFYA'] = (df['ZFYA']-df['ZFYA'].mean())/df['ZFYA'].std()
    
    le = preprocessing.LabelEncoder()
    df['race'] = le.fit_transform(df['race'])
    df['sex'] = le.fit_transform(df['sex'])
    df = df[['UGPA','LSAT','sex', 'race']]
    

    """Convert data to category"""
    for v in categorical_feature:
        df[v] = pd.Categorical(df[v].values)
        

    emb_size = 128 
    
    """Model architecture"""
    ae_model = AutoEncoder(
        encoder_layers=[512, 512, emb_size],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        activation='relu',
        swap_p=0.2,  # noise parameter
        lr=0.01,
        lr_decay=.99,
        batch_size=512,  # 512
        verbose=False,
        optimizer='sgd',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distributions
    )

    """Train model"""
    ae_model.to(device)
    ae_model.fit(df, epochs=100)

    """Save model"""
    # torch.save(ae_model, conf['ae_model_law'])
    torch.save(ae_model.state_dict(), conf['state_dict_law'])
    
    ae_model.load_state_dict(torch.load(conf['state_dict_law']))

    
    # ae_model.get_representation(df)


