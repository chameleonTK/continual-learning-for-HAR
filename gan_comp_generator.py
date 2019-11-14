from torch import nn
from torch.nn import functional as F
import utils
import torch

class Generator(nn.Module):
    def __init__(self, z_size, input_feat,
                fc_layers=3, fc_units=400, fc_drop=0, fc_bn=True, fc_nl="relu",
                gated=False, bias=True, excitability=False, excit_buffer=False):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.fc_layers = fc_layers
        self.input_feat = input_feat
        self.fc_nl = fc_nl

        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()
        inp_unit = z_size
        # self.fc1 = nn.Linear(inp_unit, fc_units, bias=bias)
        # self.fc2 = nn.Linear(fc_units, fc_units, bias=bias)
        # self.fc3 = nn.Linear(fc_units, input_feat, bias=bias)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inp_unit, fc_units, bias=True))

        for i in range(fc_layers-2):
            self.layers.append(nn.Linear(fc_units, fc_units, bias=True))
            
        self.layers.append(nn.Linear(fc_units, input_feat, bias=True))
        


    def forward(self, z):
        x = z
        for i in range(len(self.layers)-1):
            x = F.leaky_relu(self.layers[i](x))
        return F.leaky_relu(self.layers[len(self.layers)-1](x))
        # if self.fc_nl == "relu":
            
        # elif self.fc_nl == "sigmoid":
        #     return F.sigmoid(self.layers[len(self.layers)-1](x))
        # else:
        #     return self.layers[len(self.layers)-1](x)

class CondGenerator(nn.Module):
    def __init__(self, z_size, input_feat, n_classes,
                fc_layers=3, fc_units=400, fc_drop=0, fc_bn=True, fc_nl="relu",
                gated=False, bias=True, excitability=False, excit_buffer=False):
        # configurations
        super().__init__()
        self.emb = 2
        self.label_emb = nn.Embedding(n_classes, self.emb)

        self.z_size = z_size
        self.fc_layers = fc_layers
        self.input_feat = input_feat
        self.fc_nl = fc_nl
        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()

        inp_unit = z_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inp_unit+self.emb, fc_units, bias=True))

        for i in range(fc_layers-2):
            self.layers.append(nn.Linear(fc_units, fc_units, bias=True))
            
        self.layers.append(nn.Linear(fc_units, input_feat, bias=True))

    def forward(self, z, y=None):
        x = torch.cat((self.label_emb(y), z), 1)
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return F.leaky_relu(self.layers[len(self.layers)-1](x))
