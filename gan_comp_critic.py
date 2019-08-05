from torch import nn
from torch.nn import functional as F
import utils
import torch

class Critic(nn.Module):
    def __init__(self, input_feat, classes, fc_layers=3, fc_units=400, fc_drop=0, fc_bn=True, fc_nl="relu",
                gated=False, bias=True, excitability=False, excit_buffer=False):
        # configurations
        super().__init__()
        self.label = "Classifier"
        self.fc_layers = fc_layers

        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()

        # fully connected hidden layers
        inp_unit = input_feat
        self.fc1 = nn.Linear(inp_unit, fc_units, bias=bias)
        self.fc2 = nn.Linear(fc_units, fc_units, bias=bias)
        self.fc3 = nn.Linear(fc_units, classes, bias=bias)


    def forward(self, z):
        x = z
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))