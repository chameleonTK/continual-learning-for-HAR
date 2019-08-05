from functools import reduce
import torch
from torch import nn, autograd
from torch.autograd import Variable
import utils
from replayer import Replayer
from torch import optim
import numpy as np
from torch.nn import functional as F

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
        self.fc1 = nn.Linear(inp_unit+self.emb, fc_units, bias=bias)
        self.fc2 = nn.Linear(fc_units, fc_units, bias=bias)
        self.fc3 = nn.Linear(fc_units, 1, bias=bias)


    def forward(self, z, y=None):
        x = torch.cat((self.label_emb(y), z), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

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

        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()

        inp_unit = z_size
        self.fc1 = nn.Linear(inp_unit+self.emb, fc_units, bias=bias)
        self.fc2 = nn.Linear(fc_units, fc_units, bias=bias)
        self.fc3 = nn.Linear(fc_units, input_feat, bias=bias)
        self.fc_nl = fc_nl

    def forward(self, z, y=None):
        x = torch.cat((self.label_emb(y), z), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.fc_nl == "relu":
            return F.leaky_relu(self.fc3(x))
        elif self.fc_nl == "sigmoid":
            return F.sigmoid(self.fc3(x))
        else:
            return (self.fc3(x))


class CGAN(Replayer):
    def __init__(self, input_feat, n_classes = 10, cuda=False, device="cpu",
            
            z_size=20,
            critic_fc_layers=3, critic_fc_units=100, critic_lr=1e-03,
            generator_fc_layers=3, generator_fc_units=100, generator_lr=1e-03,
            generator_activation = "relu",
            critic_updates_per_generator_update=5,
            gp_lamda=10.0):

        super().__init__()
        self.label = "CGAN"
        self.z_size = z_size
        self.input_feat = input_feat

        self.cuda = cuda
        self.device = device
        self.n_classes = n_classes
        
        self.critic = CondCritic(input_feat, n_classes, fc_layers=critic_fc_layers, fc_units=critic_fc_units).to(device)
        self.generator = CondGenerator(z_size, input_feat, n_classes, fc_layers=generator_fc_layers, fc_units=generator_fc_units, fc_nl=generator_activation).to(device)

        # training related components that should be set before training.
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=0.0002, betas=(0.5, 0.9999)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=0.0002, betas=(0.5, 0.9999)
        )
        
        self.critic_updates_per_generator_update = critic_updates_per_generator_update
        self.lamda = gp_lamda

    def save_model(self, prod=False):
        if prod:
            return {
                "critic_state_dict": self.critic.state_dict(),
                "generator_state_dict": self.generator.state_dict(),
            }
            
        return {
            "critic_state_dict": self.critic.state_dict(),
            "critic_optm_state_dict": self.critic_optimizer.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "generator_optm_state_dict": self.generator_optimizer.state_dict(),
        }
    
    def load_model(self, checkpoint, class_index=None, prod=False):
        if class_index is None:
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            if not prod:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optm_state_dict"])
                self.generator_optimizer.load_state_dict(checkpoint["generator_optm_state_dict"])

        else:        
            self.critic.load_state_dict(checkpoint[str(class_index)+"_critic_state_dict"])
            self.generator.load_state_dict(checkpoint[str(class_index)+"_generator_state_dict"])
            if not prod:
                self.critic_optimizer.load_state_dict(checkpoint[str(class_index)+"_critic_optm_state_dict"])
                self.generator_optimizer.load_state_dict(checkpoint[str(class_index)+"_generator_optm_state_dict"])

                
    def forward(self, x):
        raise Exception("NO implementaion")

    def train_a_batch(self, x, y, noise=0):

        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor
        
        adversarial_loss = torch.nn.BCELoss()
        valid = Variable(FloatTensor(x.size(0), 1).fill_(1.0), requires_grad=False).to(self._device())
        fake = Variable(FloatTensor(x.size(0), 1).fill_(0.0), requires_grad=False).to(self._device())
        
        # this part adopted from https://github.com/9310gaurav/ali-pytorch/blob/master/main.py and https://github.com/soumith/ganhacks
        
        
        
        batch_size = x.shape[0]
        real_x = Variable(x.type(FloatTensor)).to(self._device())
        real_y = Variable(y.type(LongTensor)).to(self._device())

        if noise > 0:
            _n = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * noise)).to(self._device())
            real_x = real_x + _n

        self.generator_optimizer.zero_grad()

        # Sample noise as generator input
        z = self._noise(x.size(0))
        gen_y = Variable(LongTensor(np.random.randint(0, self.n_classes, batch_size))).to(self._device()) 

        # Generate a batch of images
        gen_x = self.generator(z, gen_y)
        if noise > 0:
            _n = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * noise)).to(self._device())
            gen_x = gen_x + _n

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(self.critic(gen_x, gen_y), valid)

        g_loss.backward()
        self.generator_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        for _ in range(self.critic_updates_per_generator_update):
            
            z = self._noise(x.size(0))
            gen_y = Variable(LongTensor(np.random.randint(0, self.n_classes, batch_size))).to(self._device())
    
            gen_x = self.generator(z, gen_y)
            if noise > 0:
                _n = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * noise)).to(self._device())
                gen_x = gen_x + _n
            

            self.critic_optimizer.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(self.critic(real_x, real_y), valid)
            fake_loss = adversarial_loss(self.critic(gen_x, gen_y), fake)

            d_loss = (real_loss + fake_loss)*0.5

            d_loss.backward()
            self.critic_optimizer.step()

        return {'d_cost': float(d_loss.cpu().data), 'g_cost': float(g_loss.cpu().data), 'W_dist': float(0)}

    def _noise(self, size):
        z = Variable(torch.randn(size, self.z_size)) * .1
        return z.to(self._device())

    def _device(self):
        return self.device

    def _is_on_cuda(self):
        return self.cuda
