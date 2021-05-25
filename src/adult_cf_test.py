import torch
import torch.nn as nn
import pandas as pd
import yaml
import logging
import sys
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
from model_arch.discriminator import Discriminator_Adult
from dfencoder.autoencoder import AutoEncoder
from dfencoder.dataframe import EncoderDataFrame
from utils.evaluate_func import evaluate_classifier, evaluate_distribution, evaluate_fairness
from utils.helpers import preprocess_dataset
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

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
    logger = setup_logging(conf['log_train_adult'])

    """Load data"""
    data_path = conf['processed_data_adult']
    df = pd.read_csv(data_path)

    """Setup features"""
    data_name = "adult"
    dict_ = features_setting(data_name)
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]

    df_generator = df[normal_features]

    df[target] = df[target].astype(float)

    """Preprocess data"""
    df = preprocess_dataset(df, continuous_features, categorical_features)

    """Setup auto encoder"""
    df_autoencoder = df[full_features].copy()
    emb_size = 128
    ae_model = AutoEncoder(
        input_shape=df[full_features].shape[1],
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
    ae_model.to(device)
    ae_model.build_model(df_autoencoder)
    ae_model.load_state_dict(torch.load(conf['adult_encoder']))
    ae_model.eval()

    """Logistic Regression"""
    Z = ae_model.get_representation(df_autoencoder)
    Y = np.array([float(i) for i in df[target].values])
    clf = LogisticRegression(solver='liblinear')
    clf.fit(Z.cpu().detach().numpy(), Y)
    y_pred = clf.predict(Z.cpu().detach().numpy())
    eval = evaluate_classifier(y_pred, df[target].values)


    """Setup hyperparameter"""
    parameters = {}
    parameters['epochs'] = 500
    parameters['learning_rate'] = 1e-5
    parameters['dataframe'] = df
    parameters['batch_size'] = 256
    parameters['problem'] = 'classification'
    lambda1, lambda2 = 0.5, 0.01

    """Hyperparameter"""
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    dataframe = parameters['dataframe']
    batch_size = parameters['batch_size']
    problem = parameters['problem']

    """Setup generator and discriminator"""
    emb_size = 128
    discriminator_agnostic = Discriminator_Adult(emb_size, problem)
    discriminator_agnostic.to(device)
    optimizer1 = torch.optim.Adam(
        discriminator_agnostic.parameters(), lr=learning_rate
    )


    weights = [df[target].value_counts()[0], df[target].value_counts()[1]]
    normedWeights = [1 - (x / sum(weights)) for x in weights]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    loss_fn = nn.CrossEntropyLoss(normedWeights)

    step = 0
    for i in (range(epochs)):
        df_train = df.copy()
        losses = []
        correct = 0

        """Split batch size"""
        skf = StratifiedKFold(n_splits=20, random_state=epochs, shuffle=True)
        for train_index, test_index in (skf.split(df[full_features], df[target])):
            batch_ae = df.iloc[test_index,:][full_features]
            batch_Z = ae_model.get_representation(batch_ae)
            prediction = discriminator_agnostic(batch_Z)
            Y = df.iloc[test_index,:][target].values
            Y = torch.Tensor(Y).to(device).reshape(-1,1).long()
            loss_agnostic = loss_fn(prediction, Y.reshape(-1))
            losses.append(loss_agnostic.cpu().detach().numpy())
            optimizer1.zero_grad()
            loss_agnostic.backward()
            optimizer1.step()


        df_generator = df[full_features].copy()
        """Get the final prediction"""
        Z = ae_model.get_representation(df_generator)
        y_pred = discriminator_agnostic(Z)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.reshape(-1).cpu().detach().numpy()
        y_true = df_train[target].values

        """Evaluation"""
        eval = evaluate_classifier(y_pred, y_true)
        print("Epoch {}".format(i))
        print("Loss {}".format(sum(losses)/len(losses)))
        print(eval)




