import torch
from torch.nn import functional as F
from continual_learner import ContinualLearner
from replayer import Replayer
import utils
from torch import optim
from torch import nn
import numpy as np


class Classifier(ContinualLearner, Replayer):
    # TODO: consider paper >>  icarl: Incremental classifier and representation learning.
    
    def __init__(self, input_feat, classes,
                 fc_layers=3, fc_units=1000, lr=0.001,
                 cuda=False, device="cpu"):

        # configurations
        super().__init__()
        self.classes = classes
        self.input_feat = input_feat
        self.fc_layers = fc_layers
        self.fc_units = fc_units
        self.lr = lr
        self.label = "Classifier"

        self.cuda = cuda
        self.device = device

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")


        ######------SPECIFY MODEL------######

        # flatten image to 2D-tensor
        inp_unit = input_feat
        self.flatten = utils.Flatten()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inp_unit, fc_units, bias=True))
        self.layers.append(nn.Linear(fc_units, fc_units, bias=True))
        self.layers.append(nn.Linear(fc_units, classes, bias=True))

        self.set_activation('relu')

        self.optim_list = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': lr}]
        self.optim_type = "adam"
        self.optimizer = optim.Adam(self.optim_list, betas=(0.9, 0.999))

    def add_output_units(self, n_classes):
        model = self
        newsolver = Classifier(
            self.input_feat,
            self.classes + n_classes,
            self.fc_layers,
            self.fc_units,
            self.lr,
            self.cuda,
            self.device
        ).to(self.device)

        self.classes += n_classes

        for i in range(len(newsolver.layers)):
            la = model.layers[i]
            lb = newsolver.layers[i]
            
            if la.weight.data.shape == lb.weight.data.shape:
                lb.weight.data = la.weight.data.clone().detach().requires_grad_(True).to(self.device)
                lb.bias.data = la.bias.data.clone().detach().requires_grad_(True).to(self.device)
            else:
                b0 = lb.bias.data.shape[0] - la.bias.data.shape[0]

                w0 = lb.weight.data.shape[0] - la.weight.data.shape[0]
                b = torch.zeros([b0]).to(self.device)
                w = torch.zeros([w0, lb.weight.data.shape[1]]).to(self.device)
                nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
                # kaiming_uniform_
                
                new_w = torch.cat([la.weight.data, w], dim=0).to(self.device)
                new_b = torch.cat([la.bias.data, b], dim=0).to(self.device)
                lb.weight.data = new_w.clone().detach().requires_grad_(True).to(self.device)
                lb.bias.data = new_b.clone().detach().requires_grad_(True).to(self.device)
        
        return newsolver

    def forward(self, x):
        x = self.flatten(x)
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return self.layers[len(self.layers)-1](x)
		
        # x = (self.fc1(x))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)

    def set_activation(self, nl):
        if isinstance(nl, nn.Module):
            self.nl = nl

        elif nl=="relu":
            self.nl = nn.ReLU()
        elif nl=="leakyrelu":
            self.nl = nn.LeakyReLU()
        else:
            self.nl = utils.Identity()

    @property
    def name(self):
        return ""

    def train_a_batch(self, x, y, scores=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, task=1):

        self.train()
        self.optimizer.zero_grad()

        # Train with actual x
        if x is not None:
            y_hat = self(x)
            y_hat = y_hat[:, active_classes]
            if self.distill and scores is not None:


                
                classes_per_task = int(y_hat.size(1) / task)
                y_tmp = y - (task-1)*classes_per_task

                binary_targets = np.zeros(shape=[len(y), classes_per_task], dtype='float32')
                binary_targets[range(len(y)), y_tmp.cpu()] = 1.0
                binary_targets = torch.from_numpy(binary_targets)

                binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
                binary_targets = binary_targets.to(self.device)
                #sum over classes, then average over batch
                predL = F.binary_cross_entropy_with_logits(
                    input=y_hat, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()
                loss_cur = predL

            else:
                predL = F.cross_entropy(input=y_hat, target=y)
                loss_cur = predL

            # Calculate training-precision
            # Tensor.max() return (values, indices) where indices is argmax
            # Tensor.item() is used to transform Tensor(number) => number
            precision = (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        else:
            precision = predL = None

        # Train with replayed x
        if x_ is not None:
            y_hat = self(x_)
            y_hat = y_hat[:, active_classes]

            if self.distill and scores is not None:
                log_scores_norm = F.log_softmax(y_hat / self.KD_temp, dim=1)
                targets_norm = F.softmax(scores_ / self.KD_temp, dim=1)

                n = y_hat.size(1)
                n_batch = y_hat.size(0)
                zeros_to_add = torch.zeros(n_batch, n-scores_.size(1))
                zeros_to_add = zeros_to_add.to(self.device)

                targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

                # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
                KD_loss_unnorm = -(targets_norm * log_scores_norm)
                KD_loss_unnorm = KD_loss_unnorm.sum(dim=1).mean()

                # normalize
                KD_loss = KD_loss_unnorm * (self.KD_temp**2)

                predL_r = KD_loss
                loss_replay = predL_r

                # scores_norm = torch.sigmoid(y_hat / self.KD_temp)
                # targets_norm = torch.sigmoid(scores_ / self.KD_temp)
                # KD_loss_unnorm = -( targets_norm * torch.log(scores_norm) + (1-targets_norm) * torch.log(1-scores_norm) )

            else:
                predL_r = F.cross_entropy(input=y_hat, target=y_)
                loss_replay = predL_r

        # Calculate total loss
        loss_replay = None if (x_ is None) else loss_replay

        if (x is not None) and (x_ is not None):
            loss_total = rnt*loss_cur + (1-rnt)*loss_replay
        elif x is None:
            loss_total = loss_replay
        else:
            loss_total = loss_cur

        if self.ewc:
            ewc_loss = self.ewc_loss()
            if self.ewc_lambda>0:
                loss_total += self.ewc_lambda * ewc_loss

        # Backpropagate errors
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': predL_r if (x_ is not None) else 0,
            
            'precision': precision if precision is not None else 0.,
        }


    def save_model(self, path, prod=False):
        if prod:
            model = {
                "state_dict": self.state_dict()
            }
        else:
            model = {
                "state_dict": self.state_dict(),
                "optm_state_dict": self.optimizer.state_dict()
            }

        torch.save(model, path)

    def load_model(self, path, prod=False):
        if self.cuda:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path,  map_location='cpu')

        self.load_state_dict(checkpoint["state_dict"])
        if not prod:
            self.optimizer.load_state_dict(checkpoint["optm_state_dict"])

