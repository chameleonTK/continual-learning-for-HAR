from torch import nn
from torch.nn import functional as F
import utils

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
        self.fc1 = nn.Linear(inp_unit, fc_units, bias=bias)
        self.fc2 = nn.Linear(fc_units, fc_units, bias=bias)
        self.fc3 = nn.Linear(fc_units, input_feat, bias=bias)
        


    def forward(self, z):
        x = z
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.fc_nl == "relu":
            return F.leaky_relu(self.fc3(x))
        elif self.fc_nl == "sigmoid":
            return F.sigmoid(self.fc3(x))
        else:
            return (self.fc3(x))
