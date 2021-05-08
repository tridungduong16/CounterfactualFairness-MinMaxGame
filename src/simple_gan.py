#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:51:46 2021

@author: trduong
"""

import torch.nn as nn
import yaml
import logging 
import sys
import pandas as pd
import torch

from dfencoder.autoencoder import AutoEncoder
from sklearn import preprocessing

class Generator(nn.Module):
    def __init__(self, input_length):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(input_length, input_length)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        return self.activation(self.dense_layer(x))


class Discriminator(nn.Module):
    def __init__(self, input_length):
        super(Discriminator, self).__init__()
        self.dense_layer = nn.Linear(input_length, 1)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        return self.activation(self.dense_layer(x))



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
    df = pd.read_csv(conf['data_law'])

    """Setup features"""
    categorical_features = ['race', 'sex']
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
    
    """Convert data to category"""
    for v in categorical_features:
        df[v] = pd.Categorical(df[v].values)
    
    df = df[['LSAT', 'UGPA', 'race', 'sex', 'ZFYA']]
    

    """Auto-encoder model"""
    ae_model = AutoEncoder(
        encoder_layers=[512, 512, 256],  # model architecture
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
    
    # ae_model.generator_fit(df, epochs=1)
    x = torch.randn(2, 3)
    
    
    