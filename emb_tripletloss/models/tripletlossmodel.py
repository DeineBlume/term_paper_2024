import torch
from torch import nn



if DEVICE:=os.environ.get('device'):
    DEVICE = torch.device(DEVICE)
else:
    DEVICE = torch.device('cpu')


class TripletLossModel(nn.Module):
    '''
    Invertible function interface for normalizing flow models.

    Parameters:
    -----------
    var_size: int
        Input vector size.
    '''
    def __init__(self, model = None):
        super(TripletLossModel, self).__init__()
        if model is None:
            model = nn.Sequential()
            model.add_module("l1", nn.Linear(128, 8096))
            model.add_module("a1", nn.ReLU())
            model.add_module("l3", nn.Linear(8096, 3))
            model.add_module("o", nn.Softmax(dim=-1))
            model.to(DEVICE)
        self.model = model

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    def forward(self, features):
        return self.model(features)

