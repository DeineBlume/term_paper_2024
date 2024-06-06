"""
Run the experiment with fixed random seed

This includes:
- Data preparation
- Model initialisation
- Model train
- Metrics evaluation
- Comparison of the boosting on embeddings vs ordinary boosting (plots and metrics)
"""

import pandas as pd
import random
import numpy as np
# import matplotlib.pyplot as plt
import torch
# import torch.nn.functional as F
# import torchvision
# from torchvision import transforms
from torch import nn
# from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
)
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import trange

import argparse
import sys
import re
import json

import logging
import logging.handlers
import os

logger = logging.getLogger(
    os.path.splitext(
        os.path.basename(
            sys.argv[0]
        )
    )[0]
)
device = "cuda" if torch.cuda.is_available() else "cpu"

from emb_tripletloss.models.tripletlossmodel import TripletLossModel
from emb_tripletloss.dataset.triplet_dataset import TripletDataset
import wandb

class CustomFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter): pass


def parse_args(args=sys.argv[1:]):
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=CustomFormatter)

    g = parser.add_argument_group("Check conf settings")
    g.add_argument(
        "--debug", "-d",
        action="store_true",
        default=False,
        help="Enable debugging"
    )
    g.add_argument(
        "--silent", "-s",
        action="store_true",
        default=False,
        help="Don't log to console"
    )
    g.add_argument(
        '--random-state',
        metavar="random state",
        default=42,
        type=int,
        help="Random state to set in project"
    )
    g.add_argument(
        "--train",
        metavar="train or not train",
        default=True,
        type=bool,
        help="Should we train model or not"
    )
    g.add_argument(
        "--pretrained",
        metavar="pretrained",
        default=True,
        type=bool,
        help="Should we use pretrained model or not"
    )
    g.add_argument(
        '--n-in',
        metavar="input dimention",
        default=128,
        type=int,
        help="Amount of input features"
    )
    g.add_argument(
        '--n-hidden',
        metavar="hidden dimensions",
        default=8096,
        type=int,
        help="Dimension of hidden space"
    )
    g.add_argument(
        '--n-out',
        metavar="output dimension",
        default=42,
        type=int,
        help="Amount of output features"
    )

    return parser.parse_args(args)


def setup_logging(options):
    """Configure logging."""
    root = logging.getLogger("")
    root.setLevel(logging.WARNING)
    logger.setLevel(options.debug and logging.DEBUG or logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(levelname)s[%(name)s] %(message)s"))
    root.addHandler(ch)


def data_preparation(options):
    logger.info('Preparring input data')
    data_folder = 'data/'
    file_names = os.listdir(data_folder)
    input_data = pd.concat([
        pd.read_csv(data_folder + file_name) for file_name in file_names if file_name.endswith('.csv')
    ], ignore_index=True)

    input_data.replace([np.inf, -np.inf], 3.4028237e38, inplace=True)

    feature_scaler = StandardScaler()
    target_encoder = OrdinalEncoder(max_categories=2)

    features = feature_scaler.fit_transform(input_data.drop('marker', axis=1))
    target = np.squeeze(target_encoder.fit_transform(input_data[['marker']]))

    if options.train:
        features_train, features_test, target_train, target_test = (
            train_test_split(features, target, test_size=0.3, random_state=options.random_state)
        )
        features_train = torch.tensor(features_train, device=device, dtype=torch.float)
        features_test = torch.tensor(features_test, device=device, dtype=torch.float)
        target_train = torch.tensor(target_train, device=device, dtype=torch.float)
        target_test = torch.tensor(target_test, device=device, dtype=torch.float)
        return {
            'train': {'features': features_train, 'target': target_train},
            'test': {'features': features_test, 'target': target_test}
        }
    features = torch.tensor(features, device=device, dtype=torch.float)
    target = torch.tensor(target, device=device, dtype=torch.float)
    return {'features': features, 'target': target}


def init_dataloader(options, data):
    with open('emb_tripletloss/dataset/train_dataloader_config.json', 'rb') as f:
        train_dataloader_config = json.load(f)

    if options.train:
        train_dataset = TripletDataset(data['train']['features'], data['train']['target'], train=True)
        test_dataset = TensorDataset(data['test']['features'], data['test']['target'])
        train_dataloader = DataLoader(train_dataset, **train_dataloader_config)
        test_dataloader = DataLoader(test_dataset)
        return {'train': train_dataloader, 'test': test_dataloader}
    dataset = TensorDataset(data['features'], data['target'])
    dataloader = DataLoader(dataset)
    return dataloader


def init_model(options) -> torch.nn.Module:
    if options.pretrained:
        return TripletLossModel().load_model('emb_tripletloss/models/weights_model')
    return TripletLossModel(n_in=options.n_in, n_hidden=options.n_hidden, n_out=options.n_out)


def train_model(options, model, dataloader):
    with open('emb_tripletloss/models/train_config.json', 'rb') as f:
        train_conf = json.load(f)
    optimizer = Adam(model.parameters(), **train_conf['optimizer'])
    scheduler = StepLR(optimizer, **train_conf['scheduler'])
    criterion = nn.TripletMarginLoss(**train_conf['criterion'])

    wandb.login()
    wandb.init(project="run-experiment")
    wandb.watch(model)

    train_dataloader = dataloader['train']
    train_features, train_target = train_dataloader.dataset.tensors
    train_target = train_target.cpu().numpy()

    test_dataloader = dataloader['test']
    test_features, test_target = test_dataloader.dataset.tensors
    test_target = test_target.cpu().numpy()

    classifier = GradientBoostingClassifier(max_leaf_nodes=5, n_estimators=100, max_features=20)

    num_epoch = train_conf['num_epoch']
    validate_freq = train_conf['validate_freq']
    t = trange(num_epoch)
    for epoch in t:
        t.set_description(f"Epoch {epoch}")
        running_loss = []
        for anchor_feature, positive_feature, negative_feature, anchor_label in train_dataloader:
            optimizer.zero_grad()
            anchor_out = model(anchor_feature)
            positive_out = model(positive_feature)
            negative_out = model(negative_feature)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
            wandb.log(
                {
                    "mean train loss": np.mean(running_loss)
                    #  "mean val accuracy": np.mean(val_accuracy),
                }
            )

        scheduler.step()
        if epoch % validate_freq == 0:
            with torch.no_grad():  # на валидации запрещаем фреймворку считать градиенты по параметрам
                test_embed = model(test_features).detach().cpu().numpy()
                classifier.fit(test_embed, test_target)
                test_pred = classifier.predict(test_embed)

                test_acc = accuracy_score(test_target, test_pred)
                test_f1 = f1_score(test_target, test_pred, average='weighted')
                test_roc_auc = roc_auc_score(test_target, test_pred)
                precision, recall, _ = precision_recall_curve(test_target, test_pred)
                test_pr_auc = auc(recall, precision)

                train_embed = model(train_features).detach().cpu().numpy()
                classifier.fit(train_embed, train_target)
                train_pred = classifier.predict(train_embed)

                train_acc = accuracy_score(train_target, train_pred)
                train_f1 = f1_score(train_target, train_pred, average='weighted')
                train_roc_auc = roc_auc_score(train_target, train_pred)
                precision, recall, _ = precision_recall_curve(train_target, train_pred)
                train_pr_auc = auc(recall, precision)

                wandb.log({
                    "mean test accuracy": test_acc,
                    "mean test f1": test_f1,
                    "mean test roc_auc": test_roc_auc,
                    "mean test pr_auc": test_pr_auc,

                    "mean train accuracy": train_acc,
                    "mean train f1": train_f1,
                    "mean train roc_auc": train_roc_auc,
                    "mean train pr_auc": train_pr_auc
                })

    wandb.finish()
    os.system('wandb sync --sync-all')
    return model.eval()


def make_report(options, model, dataloader): pass


if __name__ == '__main__':
    options = parse_args()
    setup_logging(options)

    logger.info('prepare data')
    data = data_preparation(options)
    if options.train:
        logger.info(f'train shapes {data["train"]["features"].shape}')
        logger.info(f'test shapes {data["test"]["features"].shape}')
    else:
        logger.info(f'shapes {data["features"]}')

    logger.info(f'initiate dataloader')
    dataloader = init_dataloader(options, data)
    logger.info('dataloader initiated')

    logger.info(f'initiate model')
    model = init_model(options)
    num_params = sum(torch.numel(p) for p in model.parameters())
    logger.info(f'model has {num_params} parameters')
    logger.info(f'model is {model}')

    if options.train:
        logger.info(f"start train")
        model = train_model(options, model, dataloader)
        logger.info(f'train finished')



