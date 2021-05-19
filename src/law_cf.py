import torch
import torch.nn as nn 
import pandas as pd
import yaml
import logging 
import sys 
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm 
from model_arch.discriminator import Discriminator_Law
from dfencoder.autoencoder import AutoEncoder
from sklearn import preprocessing
from dfencoder.dataframe import EncoderDataFrame
from utils.evaluate_func import evaluate_pred, evaluate_distribution, evaluate_fairness
from sklearn.utils import shuffle


# def train(**parameters):
#     """Hyperparameter"""
#     learning_rate = parameters['learning_rate']
#     epochs = parameters['epochs']
#     input_length = parameters['input_length']
#     dataframe = parameters['dataframe']
#
#     generator = AutoEncoder(input_length)
#     discriminator_agnostic = Discriminator_Law(input_length)
#     discriminator_awareness = Discriminator_Law(input_length)
#
#     """Optimizer"""
#     generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
#     discriminator_agnostic_optimizer = torch.optim.Adam(
#         discriminator_agnostic.parameters(), lr=learning_rate
#     )
#     discriminator_awareness_optimizer = torch.optim.Adam(
#         discriminator_awareness.parameters(), lr=learning_rate
#     )
#
#
#     """Loss function"""
#     loss = nn.BCELoss()
#
#
#     """Training steps"""
#     for i in tqdm(epochs):
#         X = dataframe['normal_features']
#         S = dataframe['sensitive_features']
#         Y = dataframe['target']
#
#         Z = generator.generator_fit(X)
#         ZS = torch.cat((Z,S),1)
#
#         predictor_agnostic = discriminator_agnostic.forward(Z)
#         predictor_awareness = discriminator_awareness.forward(Z, S)
#
#         loss_agnostic = loss(predictor_agnostic, Y)
#         loss_awareness = loss(predictor_awareness, Y)
#         final_loss = (loss_agnostic + loss_awareness) / 2
#         final_loss.backward()
#
#         generator_optimizer.step()
#         discriminator_agnostic_optimizer.step()
#         discriminator_awareness_optimizer.step()

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
    logger = logging.getLogger('CFairness')
    file_handler = logging.FileHandler(filename=conf['log_train_law'])
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
    
    df_generator = df[normal_feature]

    
    """Preprocess data"""
    df['LSAT'] = (df['LSAT']-df['LSAT'].mean())/df['LSAT'].std()
    df['UGPA'] = (df['UGPA']-df['UGPA'].mean())/df['UGPA'].std()
    df['ZFYA'] = (df['ZFYA']-df['ZFYA'].mean())/df['ZFYA'].std()
    
    le = preprocessing.LabelEncoder()
    df['race'] = le.fit_transform(df['race'])

    le = preprocessing.LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])

    df = df[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']]

    for v in categorical_feature:
        df[v] = pd.Categorical(df[v].values)

    df_autoencoder = df.copy()
    
    """Setup antuo encoder"""
    dfencoder_model = AutoEncoder(
        input_shape = df_autoencoder[full_feature].shape[1],
        encoder_layers=[512, 512, 128],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        activation='tanh',
        swap_p=0.2,  # noise parameter
        lr=0.01,
        lr_decay=.99,
        batch_size=512,  # 512
        verbose=False,
        optimizer='sgd',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distributions
    )
    
    dfencoder_model.to(device)
    dfencoder_model.fit(df_autoencoder[full_feature], epochs=100)
    
    # sys.exit(1)
    # df = pd.get_dummies(df, columns = ['sex'])
    # df = pd.get_dummies(df, columns = ['race'])

    # sensitive_feature = ['sex_0','sex_1', 'race_0', 'race_1']


    """Setup hyperparameter"""    
    logger.debug('Setup hyperparameter')
    parameters = {}
    parameters['epochs'] = 300
    parameters['learning_rate'] = 1e-9
    parameters['dataframe'] = df
    parameters['batch_size'] = 64
    parameters['problem'] = 'regression'

    """Hyperparameter"""
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    dataframe = parameters['dataframe']
    batch_size = parameters['batch_size']
    problem = parameters['problem']
    
    """Setup generator and discriminator"""
    emb_size = 64
    discriminator_agnostic = Discriminator_Law(emb_size, problem)
    discriminator_awareness = Discriminator_Law(emb_size + 4, problem)
    discriminator_agnostic.to(device)
    discriminator_awareness.to(device)

    """Setup generator"""
    df_generator = df[normal_feature]
    generator= AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[256, 256, emb_size],  # model architecture
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
    generator.build_model(df_generator)

    """Optimizer"""
    optimizer1 = torch.optim.Adam(
        generator.parameters(), lr=learning_rate
    )
    optimizer2 = torch.optim.SGD(discriminator_agnostic.parameters(),
                                 lr=learning_rate, momentum=0.9)
    optimizer3 = torch.optim.SGD(discriminator_awareness.parameters(),
                                 lr=learning_rate, momentum=0.9)

    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min')
    scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=learning_rate, max_lr=0.001)
    scheduler3 = torch.optim.lr_scheduler.CyclicLR(optimizer3, base_lr=learning_rate, max_lr=0.001)

    
    
    """Training"""
    n_updates = len(df)// batch_size
    logger.debug('Training')
    logger.debug('Number of updates {}'.format(n_updates))
    logger.debug('Dataframe length {}'.format(len(df)))
    logger.debug('Batchsize {}'.format((batch_size)))

    loss_function = torch.nn.MSELoss()
    loss_function = torch.nn.SmoothL1Loss()


    step = 0
    for i in (range(epochs)):
        df_train = df.copy().sample(frac=1).reset_index(drop=True)
        df_generator = df_train[normal_feature].copy()
        df_autoencoder = df_train.copy()

        sum_loss = []
        sum_loss_aware = []
        sum_loss_gen = []
        for j in tqdm(range(n_updates)):
            path = step % 10
            """Only contain normal features"""
            df_term_generator = df_generator.loc[batch_size*j:batch_size*(j+1)]
            df_term_generator = EncoderDataFrame(df_term_generator)
            df_term_generator_noise = df_term_generator.swap(likelihood=0.25)
            df_term_autoencoder = df_autoencoder.loc[batch_size*j:batch_size*(j+1)].reset_index(drop = True)

            """Label"""
            Y = torch.Tensor(df_term_autoencoder[target].values).to(device).reshape(-1,1)

            """Feed forward"""
            Z = generator.custom_forward(df_term_generator)
            Z_noise = generator.custom_forward(df_term_generator_noise)

            """Get the representation from autoencoder model"""
            S = dfencoder_model.get_representation(
                df_term_autoencoder[full_feature]
            )

            """Get only sensitive representation"""
            sex_feature = dfencoder_model.categorical_fts['sex']
            cats = sex_feature['cats']
            emb = sex_feature['embedding']
            cat_index = df_term_autoencoder['sex'].values
            emb_cat_sex = []
            for c in cat_index:
                emb_cat_sex.append(emb.weight.data.cpu().numpy()[cats.index(c), :].tolist())

            race_feature = dfencoder_model.categorical_fts['race']
            cats = race_feature['cats']
            emb = race_feature['embedding']
            cat_index = df_term_autoencoder['race'].values
            emb_cat_race = []
            for c in cat_index:
                emb_cat_race.append(emb.weight.data.cpu().numpy()[cats.index(c), :].tolist())

            emb_cat_race = torch.tensor(np.array(emb_cat_race).astype(np.float32)).to(device)
            emb_cat_sex = torch.tensor(np.array(emb_cat_sex).astype(np.float32)).to(device)
            emb = torch.cat((emb_cat_race, emb_cat_sex),1)

            ZS = torch.cat((emb, Z), 1)

            """Prediction and calculate loss"""
            predictor_awareness = discriminator_awareness(ZS)
            predictor_agnostic = discriminator_agnostic(Z)
            predictor_agnostic_noise = discriminator_agnostic(Z_noise)

            """Discriminator loss"""
            loss_agnostic = loss_function(predictor_agnostic, Y)
            loss_agnostic += loss_function(predictor_agnostic_noise, Y)
            loss_awareness = loss_function(predictor_awareness, Y)
            diff_loss = torch.max(torch.tensor(0).to(device), loss_agnostic - loss_awareness)

            "Generator loss"
            gen_loss = 0.1 * diff_loss + loss_agnostic

            """Track loss"""
            sum_loss.append(loss_agnostic)
            sum_loss_aware.append(loss_awareness)
            sum_loss_gen.append(gen_loss)

            """Optimizing progress"""
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            if path in [0, 1, 2]:
                gen_loss.backward()
                optimizer1.step()
                scheduler1.step(loss_agnostic)
            elif path in [3, 4, 5, 6, 7, 8]:
                loss_agnostic.backward()
                optimizer2.step()
                scheduler2.step()
            elif path in [9]:
                loss_awareness.backward()
                optimizer3.step()
                scheduler3.step()
            else:
                raise ValueError("Invalid path number. ")

            step += 1

            del df_term_generator
            del df_term_generator_noise
            del emb_cat_race
            del emb_cat_sex
            del emb
            del ZS


        df_train = df.copy()
        df_generator = df_train[normal_feature].copy()

        """Get the final prediction"""
        Z = generator.get_representation(df_generator)
        predictor_agnostic = discriminator_agnostic(Z)
        y_pred = predictor_agnostic.cpu().detach().numpy().reshape(-1)
        y_true = df_train[target].values

        """Evaluation"""
        df_result = pd.read_csv(conf['result_law'])
        df_result['inv_prediction'] = y_pred

        eval = evaluate_pred(y_pred, y_true)
        eval_fairness = evaluate_fairness(sensitive_feature, df_result, 'inv_prediction')

        """Log to file"""
        logger.debug("Epoch {}".format(i))
        logger.debug('Loss Agnostic {:.4f}'.format(sum(sum_loss)/len(sum_loss)))
        logger.debug('Loss Awareness {:.4f}'.format(sum(sum_loss_aware)/len(sum_loss)))
        logger.debug('Generator loss {:.4f}'.format(sum(sum_loss_gen)/len(sum_loss)))
        logger.debug("RMSE {:.4f}".format(eval['RMSE']))
        logger.debug("Fairness {:.7f}".format(eval_fairness['sinkhorn']))

    # print("Prediction ", y_pred)
    # print("Label ", y_true)

    """Output to file"""
    df_result = pd.read_csv(conf['result_law'])
    df_result['inv_prediction'] = y_pred
    df_result.to_csv(conf['result_law'], index = False)
    
    


    sys.modules[__name__].__dict__.clear()


    
    