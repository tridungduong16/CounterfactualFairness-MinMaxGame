import pandas as pd
import yaml
import logging
import sys
import torch

from torch import optim
from torch.utils.data import Dataset, DataLoader
from model_arch.table_architecture import load_data
from model_arch.table_architecture import DataBuilder
from model_arch.table_architecture import Autoencoder
from model_arch.table_architecture import customLoss
from model_arch.table_architecture import weights_init_uniform_rule
from model_arch.table_architecture import train
from sklearn import preprocessing


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    le = preprocessing.LabelEncoder()
    df['race'] = le.fit_transform(df['race'])

    le = preprocessing.LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])

    df = df[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']]

    for v in categorical_feature:
        df[v] = pd.Categorical(df[v].values)

    df = df.reset_index(drop=True)

    data_set = DataBuilder(df, target, full_feature)
    trainloader = DataLoader(dataset=data_set,batch_size=1024)

    D_in = data_set.x.shape[1]
    H = 50
    H2 = 12
    model = Autoencoder(D_in, H, H2).to(device)
    model.apply(weights_init_uniform_rule)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_mse = customLoss()

    epochs = 1500
    log_interval = 50
    val_losses = []
    train_losses = []



