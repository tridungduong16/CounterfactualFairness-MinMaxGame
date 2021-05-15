import torch
import torch.nn as nn 
import pandas as pd
import yaml
import logging 
import sys 
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm 
from model_arch.discriminator import Discriminator_Agnostic, Discriminator_Awareness
from dfencoder.autoencoder import AutoEncoder
from sklearn import preprocessing
from dfencoder.dataframe import EncoderDataFrame
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
from utils.evaluate_func import evaluate_pred, evaluate_distribution, evaluate_fairness
from sklearn.linear_model import LinearRegression

def train(**parameters):
    """Hyperparameter"""
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    input_length = parameters['input_length']
    dataframe = parameters['dataframe']

    generator = AutoEncoder(input_length)
    discriminator_agnostic = Discriminator_Agnostic(input_length)
    discriminator_awareness = Discriminator_Awareness(input_length)

    """Optimizer"""
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    discriminator_agnostic_optimizer = torch.optim.Adam(
        discriminator_agnostic.parameters(), lr=learning_rate
    )
    discriminator_awareness_optimizer = torch.optim.Adam(
        discriminator_awareness.parameters(), lr=learning_rate
    )


    """Loss function"""
    loss = nn.BCELoss()


    """Training steps"""
    for i in tqdm(epochs):
        X = dataframe['normal_features']
        S = dataframe['sensitive_features']
        Y = dataframe['target']

        Z = generator.generator_fit(X)

        ZS = torch.cat((Z,S),1)
        # print(ZS)
        # sys.exit(1)
        
        predictor_agnostic = discriminator_agnostic.forward(Z)
        predictor_awareness = discriminator_awareness.forward(Z, S)

        loss_agnostic = loss(predictor_agnostic, Y)
        loss_awareness = loss(predictor_awareness, Y)
        final_loss = (loss_agnostic + loss_awareness) / 2
        final_loss.backward()

        generator_optimizer.step()
        discriminator_agnostic_optimizer.step()
        discriminator_awareness_optimizer.step()

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
        encoder_layers=[512, 512, 32],  # model architecture
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
    
    dfencoder_model.to(device)
    dfencoder_model.fit(df_autoencoder[full_feature], epochs=1)    
    
    # sys.exit(1)

    # df = pd.get_dummies(df, columns = ['sex'])
    # df = pd.get_dummies(df, columns = ['race'])

    # print(df)
    # df = pd.get_dummies(df.sex, prefix='Sex')
    # sensitive_feature = ['sex_0','sex_1', 'race_0', 'race_1']
    # sys.exit(1)
    
    # X, y = df[['LSAT', 'UGPA', 'sex', 'race']].values, df['ZFYA'].values
    
    """Setup hyperparameter"""    
    logger.debug('Setup hyperparameter')
    parameters = {}
    parameters['epochs'] = 2
    parameters['learning_rate'] = 0.001
    parameters['dataframe'] = df
    parameters['batch_size'] = 32
    parameters['problem'] = 'regression'

    """Hyperparameter"""
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    dataframe = parameters['dataframe']
    batch_size = parameters['batch_size']
    problem = parameters['problem']
    
    """Setup generator and discriminator"""
    # dfencoder_model = torch.load(conf['ae_model_law'])

    emb_size = 32
    discriminator_agnostic = Discriminator_Agnostic(emb_size, problem)
    discriminator_awareness = Discriminator_Awareness(emb_size*2, problem)
    
    # generator.to(device)
    discriminator_agnostic.to(device)
    discriminator_awareness.to(device)

    """Setup generator"""
    df_generator = df[normal_feature]
    # print("Haizz", df_generator.shape[1])
    generator= AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[8, 8, emb_size],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        encoder_dropout = 0.5,
        decoder_dropout = 0.5,
        activation='leaky_relu',
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
    # generator_optimizer = torch.optim.Adam(
    #     generator.parameters(), lr=learning_rate
    # )
    # discriminator_agnostic_optimizer = torch.optim.Adam(
    #     discriminator_agnostic.parameters(), lr=learning_rate
    # )
    # discriminator_awareness_optimizer = torch.optim.Adam(
    #     discriminator_awareness.parameters(), lr=learning_rate
    # )

    optimizer1 = torch.optim.Adam((*generator.parameters(),
                                   *discriminator_agnostic.parameters()),
                                  lr=1e-6,
                                  weight_decay=1e-5)

    # optimizer1 = torch.optim.SGD((*generator.parameters(),
    #                                *discriminator_agnostic.parameters()),
    #                              lr=0.01,
    #                              momentum=0.9)

    optimizer2 = torch.optim.Adam(discriminator_awareness.parameters())

    
    # lr_decay = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, lr_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=10, gamma=0.1)
    # scheduler_discriminator_env = torch.optim.lr_scheduler.StepLR(discriminator_awareness_optimizer, step_size=10, gamma=0.1)
    # scheduler_discriminator = torch.optim.lr_scheduler.StepLR(discriminator_agnostic_optimizer, step_size=10, gamma=0.1)
    
    
    
    
    """Training"""
    n_updates = len(df)// batch_size
    logger.debug('Training')
    logger.debug('Number of updates {}'.format(n_updates))
    logger.debug('Dataframe length {}'.format(len(df)))
    logger.debug('Batchsize {}'.format((batch_size)))
    
    loss = torch.nn.SmoothL1Loss()
    loss_function = torch.nn.MSELoss()
    # loss = torch.nn.L1Loss()

    dist_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)


    
    epochs = 50
    # epochs = 1
    
    for i in (range(epochs)):
        # logger.debug('Epoch {}'.format((i)))  
        # scheduler.step()
        # scheduler_discriminator_env.step()
        # scheduler_discriminator.step()

        sum_loss = 0
        for j in tqdm(range(n_updates)):
            
            df_term_generator = df_generator.loc[batch_size*j:batch_size*(j+1)]
            df_term_generator = EncoderDataFrame(df_term_generator)
            df_term_generator_noise = df_term_generator.swap(likelihood=0.25)
            df_term_discriminator = df.loc[batch_size*j:batch_size*(j+1)].reset_index(drop = True)
            df_term_autoencoder = df_autoencoder.loc[batch_size*j:batch_size*(j+1)].reset_index(drop = True)

            X = torch.tensor(df_term_discriminator[normal_feature].values.astype(np.float32)).to(device)
            Y = torch.Tensor(df_term_discriminator[target].values).to(device).reshape(-1,1)
            Z = generator.custom_forward(df_term_generator)
            Z_noise = generator.custom_forward(df_term_generator_noise)

            S = dfencoder_model.get_representation(
                df_term_autoencoder[full_feature]
            )

            # S = torch.tensor(df_term_discriminator[sensitive_feature].values)

            ZS = torch.cat((S, Z), 1)

            predictor_awareness = discriminator_awareness(ZS)
            predictor_agnostic = discriminator_agnostic(Z)
            predictor_agnostic_noise = discriminator_agnostic(Z_noise)

            """
            loss = loss(f_i, label)
            loss = loss(f_i, f_i_noise)
            loss = loss(f_i, f_e)
            loss = loss(f_i_noise, f_e)
            """
            loss_agnostic = loss(predictor_agnostic, Y)
            loss_agnostic_noise = loss(predictor_agnostic_noise, Y)
            loss_awareness = loss(predictor_awareness, Y)

            # final_loss = loss_agnostic + loss_agnostic_noise
            final_loss = loss_agnostic + loss_agnostic_noise + \
                         0.01*F.leaky_relu(loss_agnostic + loss_agnostic_noise - loss_awareness)
            # sum_loss += loss_agnostic
            # print(loss_agnostic)
            # sum_loss += loss_agnostic_noise
            # final_loss = sum_loss
            # final_loss = loss_agnostic + loss_agnostic_noise + \
            #              0.01*F.leaky_relu(loss_agnostic - loss_awareness)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            final_loss.backward(retain_graph=True)
            for p in discriminator_awareness.parameters():
                if p.grad is not None:  # In general, C is a NN, with requires_grad=False for some layers
                    p.grad.data.mul_(-1)  # Update of grad.data not tracked in computation graph

            optimizer1.step()
            optimizer2.step()


            # generator_optimizer.zero_grad()
            # discriminator_agnostic_optimizer.zero_grad()
            # discriminator_awareness_optimizer.zero_grad()

            # loss_agnostic = loss(predictor_agnostic, Y)
            # loss_agnostic += loss(predictor_agnostic_noise, Y)
            # loss_agnostic += loss(predictor_agnostic_noise, predictor_agnostic)
            # loss_agnostic += F.relu(loss(predictor_awareness            mse, bce, cce, net_loss = self.compute_loss(
            #                 num, bin, cat, target_sample,
            #                 logging=False
            #             )
            #             self.do_backward(mse, bce, cce), predictor_agnostic))
            # loss_agnostic += F.relu(loss(predictor_awareness, predictor_agnostic_noise))
            # final_loss = loss_agnostic/5
            # num, bin, cat = generator.forward(df_term_generator)
            # mse, bce, cce, net_loss = generator.compute_loss(
            #     num, bin, cat, df_term_generator_noise,
            #     logging=False
            # )
            #
            # mse.backward(retain_graph=True)
            # bce.backward(retain_graph=True)
            # for i, ls in enumerate(cce):
            #     if i == len(cce) - 1:
            #         ls.backward(retain_graph=False)
            #     else:
            #         ls.backward(retain_graph=True)

            # generator.do_backward(mse, bce, cce)

            # mse, bce, cce, net_loss = generator.compute_loss(
            #     num, bin, cat, target_sample,
            #     logging=False
            # )
            # generator.do_backward(mse, bce, cce)

            # loss1 = loss(predictor_agnostic, Y)
            # loss1.backward(retain_graph=True)

            # loss2 = loss(predictor_agnostic_noise, Y)
            # loss2.backward(retain_graph=True)

            # loss3 = loss(predictor_agnostic, predictor_agnostic_noise)
            # loss3.backward(retain_graph=True)

            # loss4 = F.relu(loss1 - loss3)
            # loss4.backward(retain_graph=True)

            # loss5 = F.relu(loss2 - loss3)
            # loss5.backward(retain_graph=True)

            # final_loss = loss1 + loss2 + loss3 + 0.1*loss4 + 0.1*loss5
            # loss3 += loss(predictor_agnostic_noise, predictor_agnostic)
            # loss4 += F.relu(loss(predictor_awareness, predictor_agnostic))
            # loss5 += F.relu(loss(predictor_awareness, predictor_agnostic_noise))


            # final_loss = loss_agnostic + F.leaky_relu(loss_agnostic - torch.sigmoid(loss_awareness))
            # final_loss = 100*loss_agnostic + 0.0001*F.leaky_relu(loss_agnostic - loss_awareness)
            # final_loss = loss_agnostic + F.relu(loss_agnostic - loss_awareness)
            # final_loss = loss_agnostic + F.gelu(loss_agnostic - loss_awareness)
            # final_loss = loss_agnostic + F.prelu(loss_agnostic - loss_awareness, torch.tensor(0.5).to(device))
            # final_loss = loss_agnostic + F.rrelu(loss_agnostic - loss_awareness)

            
            

            
            # final_loss.backward(retain_graph=True)

            # generator_optimizer.step()
            # discriminator_agnostic_optimizer.step()
            # discriminator_awareness_optimizer.step()

            # scheduler.step()
            # scheduler_discriminator_env.step()
            # scheduler_discriminator.step()

        # if i % 10 == 0:
        logger.debug('Epoch {}'.format(i))
        logger.debug("Sum Loss {}".format(sum_loss / n_updates))
        # logger.debug('Loss Agnostic {}'.format(loss_agnostic))
        logger.debug('-------------------')

        S = dfencoder_model.get_representation(
            df_autoencoder[full_feature]
        )

        predictor_agnostic = discriminator_agnostic(S)
        logger.debug("Prediction")

        # S = S.cpu().detach().numpy()
        # reg = LinearRegression().fit(S, df_autoencoder[target].values)
        # y_pred = reg.predict(S).reshape(-1)
        y_pred = predictor_agnostic.cpu().detach().numpy().reshape(-1)
        y_true = df_autoencoder[target].values
        eval = evaluate_pred(y_pred, y_true)
        print(y_pred)
        print(eval)

    df_result = pd.read_csv(conf['result_law'])
    df_result['inv_prediction'] = y_pred
    df_result.to_csv(conf['result_law'], index = False)
    
    


    sys.modules[__name__].__dict__.clear()


    
    