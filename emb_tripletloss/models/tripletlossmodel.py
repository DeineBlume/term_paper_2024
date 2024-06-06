from collections.abc import Iterable

import torch
from torch import nn
import os

if DEVICE := os.environ.get('device'):
    DEVICE = torch.device(DEVICE)
else:
    DEVICE = torch.device('cpu')


class TripletLossModel(nn.Module):
    def __init__(self, model=None, n_in=128, n_hidden=8096, n_out=3):
        super(TripletLossModel, self).__init__()
        if model is None:
            if isinstance(n_hidden, int):
                model = nn.Sequential()
                model.add_module("input_layer", nn.Linear(n_in, n_hidden))
                model.add_module("input_activation", nn.ReLU())
                model.add_module("output_layer", nn.Linear(n_hidden, n_out))
                model.add_module("output_activation", nn.Softmax(dim=-1))
                model.to(DEVICE)
            if isinstance(n_hidden, Iterable):
                model = nn.Sequential()
                for i, n_hidden_i in enumerate(n_hidden):
                    if i == 0:
                        model.add_module("input_layer", nn.Linear(n_in, n_hidden_i))
                        model.add_module("input_activation", nn.ReLU())
                    else:
                        model.add_module(f"hidden_layer_{i}", nn.Linear(n_hidden_prev, n_hidden_i))
                        model.add_module(f"hidden_activation_{i}", nn.ReLU())
                    n_hidden_prev = n_hidden_i
                model.add_module("output_layer", nn.Linear(n_hidden_prev, n_out))
                model.add_module("output_activation", nn.Softmax(dim=-1))
        self.model = model

    def load_model(self, model_path):
        self.model = nn.Sequential()
        self.model.add_module('l1', nn.Linear(128, 8096))
        self.model.add_module('a1', nn.ReLU())
        self.model.add_module('l3', nn.Linear(8096, 3))
        self.model.add_module('o', nn.Softmax(dim=-1))
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        return self

    def forward(self, features):
        return self.model(features)

