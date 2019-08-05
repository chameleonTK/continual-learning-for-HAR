from functools import reduce
import torch
from torch import nn, autograd
from torch.autograd import Variable
import utils
from replayer import Replayer
from gan_comp_critic import Critic
from gan_comp_generator import Generator
from torch import optim

EPSILON = 1e-16
class WGAN(Replayer):
    def __init__(self, input_feat, cuda=False, device="cpu",
            
            z_size=20,
            critic_fc_layers=3, critic_fc_units=100, critic_lr=1e-03,
            generator_fc_layers=3, generator_fc_units=100, generator_lr=1e-03,
            generator_activation = "relu",
            critic_updates_per_generator_update=5,
            gp_lamda=10.0):

        super().__init__()
        self.label = "WGAN"
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
            lr=critic_lr, weight_decay=1e-05,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=generator_lr, weight_decay=1e-05,
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
        raise("NO implementaion")

    def train_a_batch(self, x, noise=0):

        # Code from https://github.com/caogang/wgan-gp
        one = torch.tensor(1.0)
        mone = torch.tensor(-1.0)
        
        one = one.to(self._device())
        mone = mone.to(self._device())

        if noise > 0:
            _n = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * noise)).to(self._device())
            x = x + _n

        for _ in range(self.critic_updates_per_generator_update):
            self.critic_optimizer.zero_grad()
            

            # train with real
            D_real = self.critic(x)
            
            
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            z = self._noise(x.size(0))
            g = self.generator(z)

            if noise > 0:
                _n = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * noise)).to(self._device())
                g = g + _n

            D_fake = self.critic(g)
            
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = self._gradient_penalty(x, g, self.lamda)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            self.critic_optimizer.step()


        # run the generator and backpropagate the errors.
        self.generator_optimizer.zero_grad()

        
        z = self._noise(x.size(0))

        fake = self.generator(z)
        if noise > 0:
            _n = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * noise)).to(self._device())
            fake = fake + _n
            
        G = self.critic(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        self.generator_optimizer.step()
        

        return {'d_cost': float(D_cost.cpu().data), 'g_cost': float(G_cost.cpu().data), 'W_dist': float(Wasserstein_D.cpu().data)}

    def _noise(self, size):
        z = Variable(torch.randn(size, self.z_size)) * .1
        return z.to(self._device())

    def _gradient_penalty(self, x, g, lamda):
        assert x.size() == g.size()
        a = torch.rand(x.size(0), 1)
        a = a.cuda() if self._is_on_cuda() else a
        
        a = a.expand(x.size(0), x.nelement()//x.size(0)).contiguous()
        interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)
        c = self.critic(interpolated)
        gradients = autograd.grad(
            c, interpolated, grad_outputs=(
                torch.ones(c.size()).cuda() if self._is_on_cuda() else
                torch.ones(c.size())
            ),
            create_graph=True,
            retain_graph=True,
        )[0]
        return lamda * ((1-(gradients+EPSILON).norm(2, dim=1))**2).mean()

    def _device(self):
        return self.device

    def _is_on_cuda(self):
        return self.cuda
