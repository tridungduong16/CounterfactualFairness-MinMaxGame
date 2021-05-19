import torch
import torch.nn as nn 
import pandas as pd
import yaml
import logging 
import sys 
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb

from tqdm import tqdm 
from model_arch.discriminator import Discriminator_Agnostic, Discriminator_Awareness
from dfencoder.autoencoder import AutoEncoder
from sklearn import preprocessing
from dfencoder.dataframe import EncoderDataFrame
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
from utils.evaluate_func import evaluate_classifier, evaluate_distribution, evaluate_fairness
from sklearn.linear_model import LogisticRegression

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
        print(ZS)
        sys.exit(1)    
        
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
    # df = df.sample(frac=0.2, replace=True, random_state=1).reset_index(drop=True)

    """Setup features"""
    categorical_features = ['marital_status', 'occupation', 'race', 'gender', 'workclass', 'education']
    continuous_features = ['age', 'hours_per_week']
    normal_features = ['age', 'workclass', 'marital_status', 'occupation', 'hours_per_week', 'education']
    full_features = ['age', 'workclass', 'education', 'marital_status', 'occupation', 'hours_per_week', 'race',
                     'gender']
    sensitive_features = ['race', 'gender']
    target = 'income'

    df_generator = df[normal_features]

    
    """Preprocess data"""
    for c in continuous_features:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    count_dict = {}

    for c in categorical_features:
        count_dict[c] = len(df[c].unique())
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c])

    for v in categorical_features:
        df[v] = pd.Categorical(df[v].values)
    
    df_autoencoder = df.copy()
    print("Input shape {}".format(df_autoencoder[full_features].shape[1]))
    print(df['income'].value_counts())
    # print(df_autoencoder)
    # sys.exit(1)

    """Setup antuo encoder"""
    dfencoder_model = AutoEncoder(
        input_shape = df_autoencoder[full_features].shape[1],
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
    
    dfencoder_model.to(device)
    dfencoder_model.fit(df_autoencoder[full_features], epochs=25)

    # sys.exit(1)

    """Setup hyperparameter"""    
    logger.debug('Setup hyperparameter')
    parameters = {}
    parameters['epochs'] = 1000
    parameters['learning_rate'] = 1e-9
    parameters['dataframe'] = df
    parameters['batch_size'] = 256
    parameters['problem'] = 'classification'

    """Hyperparameter"""
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    dataframe = parameters['dataframe']
    batch_size = parameters['batch_size']
    problem = parameters['problem']
    
    """Setup generator and discriminator"""
    # dfencoder_model = torch.load(conf['ae_model_law'])

    emb_size = 64
    print("Input shape ", df[normal_features].shape[1])
    # generator = Generator(df_generator.shape[1])
    generator= AutoEncoder(
        input_shape=df[normal_features].shape[1],
        encoder_layers=[64, 64, emb_size],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        encoder_dropout = 0.85,
        decoder_dropout = 0.85,
        activation='relu',
        swap_p=0.2,  # noise parameter
        lr=0.001,
        lr_decay=.99,
        batch_size=512,  # 512
        verbose=False,
        optimizer='adamW',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distributions
    )        
    discriminator_agnostic = Discriminator_Agnostic(emb_size, problem)
    # discriminator_awareness = Discriminator_Awareness(emb_size+len(sensitive_feature), problem)
    discriminator_awareness = Discriminator_Agnostic(emb_size+4, problem)
    
    # generator.to(device)
    discriminator_agnostic.to(device)
    discriminator_awareness.to(device)

    """Setup generator"""
    df_generator = df[normal_features].copy()

    generator.build_model(df_generator)

    """Optimizer"""
    optimizer1 = torch.optim.Adam(
        generator.parameters(), lr=learning_rate
    )
    optimizer2 = torch.optim.SGD(discriminator_agnostic.parameters(),lr=learning_rate, momentum=0.9)
    optimizer3 = torch.optim.SGD(discriminator_awareness.parameters(),lr=learning_rate, momentum=0.9)

    # optimizer2 = torch.optim.Adam(
    #     discriminator_agnostic.parameters(), lr=learning_rate
    # )
    # scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min')
    # scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=learning_rate, max_lr=0.1)
    # scheduler3 = torch.optim.lr_scheduler.CyclicLR(optimizer3, base_lr=learning_rate, max_lr=0.1)
    #

    
    
    """Training"""
    n_updates = len(df)// batch_size
    logger.debug('Training')
    logger.debug('Number of updates {}'.format(n_updates))
    logger.debug('Dataframe length {}'.format(len(df)))
    logger.debug('Batchsize {}'.format((batch_size)))
    
    # loss = torch.nn.SmoothL1Loss()
    # loss = torch.nn.MSELoss()
    # loss = torch.nn.L1Loss
    # loss = torch.nn.CrossEntropyLoss()
    # loss = torch.nn.BCELoss()

    weights = [4866, 1646]
    normedWeights = [1 - (x / sum(weights)) for x in weights]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    loss = nn.CrossEntropyLoss(weight=normedWeights)
    # loss = nn.CrossEntropyLoss()
    dist_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)


    

    step = 0
    for i in (range(epochs)):

        sum_loss = 0
        sum_loss_aware = 0
        sum_loss_gen = 0

        for j in tqdm(range(n_updates)):
            path = step % 3

            df_term_generator = df_generator.loc[batch_size*j:batch_size*(j+1)]
            df_term_generator = EncoderDataFrame(df_term_generator)
            df_term_generator_noise = df_term_generator.swap(likelihood=0.2)

            df_term_discriminator = df.loc[batch_size*j:batch_size*(j+1)].reset_index(drop = True)
            df_term_autoencoder = df_autoencoder.loc[batch_size*j:batch_size*(j+1)].reset_index(drop = True)

            # X = torch.tensor(df_term_discriminator[normal_features].values.astype(np.float32)).to(device)
            Y = torch.Tensor(df_term_discriminator[target].values).to(device).long()
            Z = generator.custom_forward(df_term_generator)
            Z_noise = generator.custom_forward(df_term_generator_noise)

            # S = dfencoder_model.get_representation(
            #     df_term_autoencoder[full_features]
            # )
            """Get only sensitive representation"""
            sex_feature = dfencoder_model.categorical_fts['gender']
            cats = sex_feature['cats']
            emb = sex_feature['embedding']
            cat_index = df_term_autoencoder['gender'].values
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
            # print(ZS.shape)

            predictor_awareness = discriminator_awareness(ZS)
            predictor_agnostic = discriminator_agnostic(Z)
            predictor_agnostic_noise = discriminator_agnostic(Z_noise)

            # print(predictor_agnostic.shape)
            # print(Y)
            loss_agnostic = loss(predictor_agnostic, Y)
            loss_agnostic += loss(predictor_agnostic_noise, Y)
            loss_awareness = loss(predictor_awareness, Y)
            diff_loss = torch.max(torch.tensor(0).to(device), loss_agnostic - loss_awareness)
            # diff_loss = torch.nn.functional.leaky_relu(loss_agnostic - loss_awareness)
            #

            gen_loss = 0.1 * diff_loss + loss_agnostic

            sum_loss += loss_agnostic
            sum_loss_aware += loss_awareness
            sum_loss_gen += gen_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            # loss_agnostic.backward()
            # optimizer2.step()
            # scheduler2.step(loss_agnostic)

            # if path in [0, 9, 6, 7]:
            if path in [0]:
                gen_loss.backward()
                optimizer1.step()
                # scheduler1.step(loss_agnostic)
            # elif path in [1, 2, 3]:
            elif path in [1]:
                loss_agnostic.backward()
                optimizer2.step()
                # scheduler2.step()
            # elif path in [8, 4, 5]:
            elif path in [2]:
                loss_awareness.backward()
                optimizer3.step()
                # scheduler3.step()

            else:
                raise ValueError("Invalid path number. ")

            step += 1

        Z = generator.get_representation(df_generator)

        # Z = dfencoder_model.get_representation(df_autoencoder[full_features])
        # Z = dfencoder_model.get_representation(
        #         df_autoencoder[full_features]
        #     ).cpu().detach().numpy()

        # Z = np.array(Z)
        # print(Z.shape)
        # print(df_autoencoder[target].values.shape)
        # print(df_autoencoder[target].dtypes)
        # clf = LogisticRegression()
        # clf = lgb.LGBMClassifier()
        # clf.fit(Z, df_autoencoder[target].values.reshape(-1,1))
        # y_pred = clf.predict(Z)
        # y_pred_prob = clf.predict_proba(Z)[:, 0]

        predictor_agnostic = discriminator_agnostic(Z)
        y_pred = torch.argmax(predictor_agnostic, dim=1).cpu().detach().numpy().reshape(-1)
        y_pred_prob = predictor_agnostic.cpu().detach().numpy()[:, 0]
        y_true = df_autoencoder[target].values

        eval = evaluate_classifier(y_pred, y_true)
        logger.debug("Epoch {}".format(i))
        logger.debug('Loss Agnostic {}'.format(sum_loss/n_updates))
        logger.debug('Loss Awareness {}'.format(sum_loss_aware/n_updates))
        logger.debug('Loss Generator {}'.format(sum_loss_aware/n_updates))
        logger.debug("Prediction")
        logger.debug(y_pred)
        logger.debug("Evaluation")
        logger.debug("Accuracy {:.4f}".format(eval['Accuracy']))
        logger.debug("Precision {:.4f}".format(eval['Precision']))
        logger.debug("Recall {:.4f}".format(eval['Recall']))
        logger.debug("Ratio {:.4f}".format(np.count_nonzero(y_pred == 0) / len(y_pred)))

        
    df_result = pd.read_csv(conf['result_adult'])
    df_result['inv_prediction'] = y_pred
    df_result['inv_prediction_proba'] = y_pred_prob
    df_result.to_csv(conf['result_adult'], index = False)
    
    print(df_result)


    sys.modules[__name__].__dict__.clear()
    



    
    