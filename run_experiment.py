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
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch import nn
from tqdm.notebook import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from torch.optim import lr_scheduler
import wandb
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

class CustomFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter): pass


# class QualityException(Exception):
#     """Raised when the input json do not pass quality assertions."""
#
#     def __init__(self, amount_of_errors=1):
#         self.message = f"{amount_of_errors} errors in total. Please correct them and push json later."
#         super().__init__(self.message)


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
        help="enable debugging"
    )
    g.add_argument(
        "--silent", "-s",
        action="store_true",
        default=False,
        help="don't log to console"
    )
    g.add_argument(
        '--random-state',
        metavar="random state",
        default=42,
        type=int,
        help="Random state to set in project"
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

    features_train, features_test, target_train, target_test = (
        train_test_split(features, target, test_size=0.3, random_state=options.)
    )

if __name__ == '__main__':
    pass