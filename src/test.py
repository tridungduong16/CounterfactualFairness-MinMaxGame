import torch
import pandas as pd
import sys
import numpy as np

from tqdm import tqdm
from model_arch.discriminator import Discriminator_Law
from dfencoder.autoencoder import AutoEncoder
from dfencoder.dataframe import EncoderDataFrame
from utils.evaluate_func import evaluate_pred, evaluate_distribution, evaluate_fairness
from utils.helpers import preprocess_dataset
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting
from sklearn.linear_model import LinearRegression

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

    """Set up logging"""
    logger = setup_logging(conf['log_train_law'])

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
    df = df.reset_index(drop=True)

    """Preprocess data"""
    df = preprocess_dataset(df, continuous_features, categorical_features)
    df['ZFYA'] = (df['ZFYA'] - df['ZFYA'].mean()) / df['ZFYA'].std()
    df = df[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']]

    df_autoencoder = df[full_features].copy()

    ae_model = AutoEncoder(
        input_shape=df[full_features].shape[1],
        encoder_layers=[512, 512, 128],  # model architecture
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
    ae_model.to(device)
    ae_model.build_model(df[full_features].copy())
    ae_model.load_state_dict(torch.load(conf['law_encoder']))
    ae_model.eval()

    df_generator = df[normal_features]
    generator= AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[256, 256, 64],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        encoder_dropout = 0.5,
        decoder_dropout = 0.5,
        activation='tanh',
        swap_p=0.2,  # noise parameter
        lr=0.0001,
        lr_decay=.99,
        batch_size=512,  # 512
        verbose=False,
        optimizer='sgd',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distributions
    )
    generator.to(device)
    generator.build_model(df_generator)
    generator.load_state_dict(torch.load(conf['law_generator']))
    generator.eval()

    Z = ae_model.get_representation(
        df[full_features]
    ).cpu().detach().numpy()

    Z_fair = generator.get_representation(
        df[normal_features]
    ).cpu().detach().numpy()

    reg = LinearRegression().fit(Z, df['ZFYA'])
    reg_fair = LinearRegression().fit(Z_fair, df['ZFYA'])

    df = pd.read_csv(conf['result_law'])

    y_pred = reg.predict(Z)
    df['full_prediction'] = y_pred.reshape(-1)

    y_pred = reg_fair.predict(Z_fair)
    df['unaware_prediction'] = y_pred.reshape(-1)

    df.to_csv(conf['result_law'], index=False)