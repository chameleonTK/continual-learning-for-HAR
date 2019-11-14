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
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inp_unit, fc_units, bias=True))

        for i in range(fc_layers-2):
            self.layers.append(nn.Linear(fc_units, fc_units, bias=True))
            
        self.layers.append(nn.Linear(fc_units, 1, bias=True))



    def forward(self, z):
        x = z
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return torch.sigmoid(self.layers[len(self.layers)-1](x))



class CondCritic(nn.Module):
    def __init__(self, input_feat, n_classes, fc_layers=3, fc_units=400, fc_drop=0, fc_bn=True, fc_nl="relu",
                gated=False, bias=True, excitability=False, excit_buffer=False):
        # configurations
        super().__init__()
        self.label = "Classifier"
        self.emb = 2
        self.label_emb = nn.Embedding(n_classes, self.emb)

        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()

        # fully connected hidden layers
        inp_unit = input_feat
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inp_unit+self.emb, fc_units, bias=True))

        for i in range(fc_layers-2):
            self.layers.append(nn.Linear(fc_units, fc_units, bias=True))
            
        self.layers.append(nn.Linear(fc_units, 1, bias=True))

        


    def forward(self, z, y=None):
        x = torch.cat((self.label_emb(y), z), 1)
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return torch.sigmoid(self.layers[len(self.layers)-1](x))