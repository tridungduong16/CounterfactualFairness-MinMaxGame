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
from sklearn.model_selection import StratifiedKFold


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
    ae_model.build_model(df[full_features].copy())
    ae_model.load_state_dict(torch.load(conf['adult_encoder']))
    ae_model.eval()


    """Setup hyperparameter"""    
    logger.debug('Setup hyperparameter')
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
    discriminator_awareness = Discriminator_Adult(emb_size + 4, problem)
    discriminator_agnostic.to(device)
    discriminator_awareness.to(device)

    """Setup generator"""
    df_generator = df[normal_features]
    generator= AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[16, 16, emb_size],  # model architecture
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
        generator.parameters(), lr=learning_rate, weight_decay=1e-10
    )

    # optimizer2 = torch.optim.AdamW(discriminator_agnostic.parameters(),
    #                                lr=learning_rate,
    #                                betas=(0.9, 0.999),
    #                                eps=1e-08,
    #                                weight_decay=0.01, amsgrad=False)


    optimizer2 = torch.optim.SGD(discriminator_agnostic.parameters(),
                                 lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer3 = torch.optim.SGD(discriminator_awareness.parameters(),
                                 lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min')
    scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=learning_rate, max_lr=0.001)
    scheduler3 = torch.optim.lr_scheduler.CyclicLR(optimizer3, base_lr=learning_rate, max_lr=0.001)

    """Training"""
    n_updates = len(df)// batch_size
    logger.debug('Training')
    logger.debug('Number of updates {}'.format(n_updates))
    logger.debug('Dataframe length {}'.format(len(df)))
    logger.debug('Batchsize {}'.format((batch_size)))

    # weights = [4866, 1646]
    weights = [df[target].value_counts()[0], df[target].value_counts()[1]]
    # print(df[target].value_counts()[0])
    normedWeights = [1 - (x / sum(weights)) for x in weights]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    loss_fn = nn.CrossEntropyLoss(normedWeights)


    step = 0
    for i in (range(epochs)):
        df_train = df.copy().sample(frac=1).reset_index(drop=True)


        sum_loss = []
        sum_loss_aware = []
        sum_loss_gen = []

        """Split batch size"""
        skf = StratifiedKFold(n_splits=20, random_state=epochs, shuffle=True)
        for train_index, test_index in (skf.split(df_train[full_features], df_train[target])):
            path = step % 10
            batch_generator = df_train.iloc[test_index, :][normal_features].copy()
            batch_generator = EncoderDataFrame(batch_generator)
            batch_generator_noise = batch_generator.swap(likelihood=0.2)
            batch_encoder = df_train.iloc[test_index, :][full_features].copy()
            Y = torch.Tensor(df_train.iloc[test_index, :][target].values).to(device).long()

            """Feed forward"""
            Z = generator.custom_forward(batch_generator)
            Z_noise = generator.custom_forward(batch_generator_noise)

            """Get the representation from autoencoder model"""
            S = ae_model.get_representation(
                batch_encoder
            )

            """Get only sensitive representation"""
            sex_feature = ae_model.categorical_fts['gender']
            cats = sex_feature['cats']
            emb = sex_feature['embedding']
            cat_index = batch_encoder['gender'].values
            emb_cat_sex = []
            for c in cat_index:
                emb_cat_sex.append(emb.weight.data.cpu().numpy()[cats.index(c), :].tolist())

            race_feature = ae_model.categorical_fts['race']
            cats = race_feature['cats']
            emb = race_feature['embedding']
            cat_index = batch_encoder['race'].values
            emb_cat_race = []
            for c in cat_index:
                emb_cat_race.append(emb.weight.data.cpu().numpy()[cats.index(c), :].tolist())

            emb_cat_race = torch.tensor(np.array(emb_cat_race).astype(np.float32)).to(device)
            emb_cat_sex = torch.tensor(np.array(emb_cat_sex).astype(np.float32)).to(device)
            emb = torch.cat((emb_cat_race, emb_cat_sex),1)

            """Concat generator and sensitive representation"""
            ZS = torch.cat((emb, Z), 1)

            """Prediction and calculate loss"""
            predictor_awareness = discriminator_awareness(ZS)
            predictor_agnostic = discriminator_agnostic(S)
            predictor_agnostic_noise = discriminator_agnostic(Z_noise)

            """Discriminator loss"""
            loss_agnostic = loss_fn(predictor_agnostic, Y.reshape(-1))
            # loss_agnostic += loss_fn(predictor_agnostic_noise, Y.reshape(-1))
            # loss_awareness = loss_fn(predictor_awareness, Y.reshape(-1))
            # diff_loss = torch.max(torch.tensor(0).to(device), loss_agnostic - loss_awareness)

            "Generator loss"
            # gen_loss = 0.01 * diff_loss + loss_agnostic

            """Track loss"""
            sum_loss.append(loss_agnostic)
            # sum_loss_aware.append(loss_awareness)
            # sum_loss_gen.append(gen_loss)

            optimizer2.zero_grad()
            loss_agnostic.backward()
            optimizer2.step()

            """Optimizing progress"""
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            # optimizer3.zero_grad()
            #
            # for p in discriminator_awareness.parameters():
            #     if p.grad is not None:  # In general, C is a NN, with requires_grad=False for some layers
            #         p.grad.data.mul_(-1)  # Update of grad.data not tracked in computation graph
            #
            # if path in [0, 1]:
            #     gen_loss.backward()
            #     optimizer1.step()
            #     # scheduler1.step(gen_loss)
            # elif path in [2, 3, 4, 5, 6, 7]:
            #     loss_agnostic.backward()
            #     optimizer2.step()
            #     # scheduler2.step()
            # elif path in [8, 9]:
            #     loss_awareness.backward()
            #     optimizer3.step()
            #     # scheduler3.step()
            # else:
            #     raise ValueError("Invalid path number. ")

            step += 1

            del batch_generator
            del batch_generator_noise
            del emb_cat_race
            del emb_cat_sex
            del emb
            del ZS

        df_generator = df_train[normal_features].copy()

        """Get the final prediction"""
        Z = generator.get_representation(df_generator)
        y_pred = discriminator_agnostic(Z)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.reshape(-1).cpu().detach().numpy()
        y_true = df_train[target].values

        """Evaluation"""
        eval = evaluate_classifier(y_pred, y_true)

        """Log to file"""
        logger.debug("Epoch {}".format(i))
        logger.debug('Loss Agnostic {:.4f}'.format(sum(sum_loss)/len(sum_loss)))
        logger.debug('Loss Awareness {:.4f}'.format(sum(sum_loss_aware)/len(sum_loss)))
        logger.debug('Generator loss {:.4f}'.format(sum(sum_loss_gen)/len(sum_loss)))
        logger.debug("F1 Score {:.4f}".format(eval['F1 Score']))
        logger.debug("Precision {:.4f}".format(eval['Precision']))
        logger.debug("Recall {:.4f}".format(eval['Recall']))
        logger.debug("Accuracy {:.4f}".format(eval['Accuracy']))
        logger.debug("-------------------------------------------")

        # logger.debug("Fairness {:.7f}".format(eval_fairness['sinkhorn']))

    """Save model"""
    logger.debug("Saving model......")
    torch.save(generator.state_dict(), conf["adult_generator"])
    torch.save(discriminator_agnostic.state_dict(), conf["adult_discriminator"])

    """Output to file"""
    # logger.debug("Output to file......")
    # df_result = pd.read_csv(conf['result_adult'])
    # df_result['inv_prediction'] = y_pred
    # df_result['inv_prediction_proba'] = y_pred_prob
    # df_result.to_csv(conf['result_adult'], index = False)

    sys.modules[__name__].__dict__.clear()
    



    
    