from functools import reduce
import torch
from torch import nn, autograd
from torch.autograd import Variable
import utils
from replayer import Replayer
from gan_comp_critic import Critic
from gan_comp_generator import Generator
from torch import optim
import numpy as np

EPSILON = 1e-16
class GAN(Replayer):
    def __init__(self, input_feat, cuda=False, device="cpu",
            
            z_size=20,
            critic_fc_layers=3, critic_fc_units=100, critic_lr=1e-03,
            generator_fc_layers=3, generator_fc_units=100, generator_lr=1e-03,
            generator_activation = "relu",

            critic_updates_per_generator_update=5,
            gp_lamda=10.0):

        super().__init__()
        self.label = "GAN"
        self.z_size = z_size
        self.input_feat = input_feat

        self.cuda = cuda
        self.device = device
        
        # components
        self.critic = Critic(input_feat, 1, fc_layers=critic_fc_layers, fc_units=critic_fc_units).to(device)
        self.generator = Generator(z_size, input_feat, fc_layers=generator_fc_layers, fc_units=generator_fc_units, fc_nl=generator_activation).to(device)

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

    def train_a_batch(self, x, noise=0):
        Tensor = torch.FloatTensor
        adversarial_loss = torch.nn.BCELoss()
        valid = Variable(Tensor(x.size(0), 1).fill_(0.9), requires_grad=False).to(self._device())
        fake = Variable(Tensor(x.size(0), 1).fill_(0.1), requires_grad=False).to(self._device())
        

        

        real_x = Variable(x.type(Tensor)).to(self._device())
        if noise > 0:
            _n = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * noise)).to(self._device())
            real_x = real_x + _n

        self.generator_optimizer.zero_grad()

        # Sample noise as generator input
        z = self._noise(x.size(0))

        # Generate a batch of images
        gen_x = self.generator(z)
        if noise > 0:
            _n = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * noise)).to(self._device())
            gen_x = gen_x + _n

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(self.critic(gen_x), valid)

        g_loss.backward()
        self.generator_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        for _ in range(self.critic_updates_per_generator_update):
            
            z = self._noise(x.size(0))
            gen_x = self.generator(z)
            if noise > 0:
                _n = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * noise)).to(self._device())
                gen_x = gen_x + _n
                
            self.critic_optimizer.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(self.critic(real_x), valid)
            fake_loss = adversarial_loss(self.critic(gen_x.detach()), fake)

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
