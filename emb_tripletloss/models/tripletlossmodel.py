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
            model = nn.Sequential()
            model.add_module("l1", nn.Linear(n_in, n_hidden))
            model.add_module("a1", nn.ReLU())
            model.add_module("l3", nn.Linear(n_hidden, n_out))
            model.add_module("o", nn.Softmax(dim=-1))
            model.to(DEVICE)
        self.model = model

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        return self

    def forward(self, features):
        return self.model(features)

