from functools import reduce
import torch
from torch import nn, autograd
from torch.autograd import Variable
import utils
from replayer import Replayer
from model_wgan import WGAN
from model_gan import GAN
import numpy as np

import tqdm

EPSILON = 1e-16
class GeneratorMultipleGAN(Replayer):
    def __init__(self, input_feat, cuda=False, device="cpu",
            model="wgan", z_size=20,
            critic_fc_layers=3, critic_fc_units=100, critic_lr=1e-03,
            generator_fc_layers=3, generator_fc_units=100, generator_lr=1e-03,
            critic_updates_per_generator_update=5, generator_activation="relu",
            gp_lamda=10.0):
            
        super().__init__()
        self.label = "Generator with multiple GANs"
        self.input_feat = input_feat
        self.model = model
        self.cuda = cuda
        self.device = device
        self.classes = 0
        
        self.generators = {}

        self.critic_fc_units = critic_fc_units
        self.critic_fc_layers = critic_fc_layers

        self.generator_fc_units = generator_fc_units
        self.generator_fc_layers = generator_fc_layers

        self.generator_activation = generator_activation
        self.noisy = False

    def get_model(self):
        if self.model == "gan":
            return GAN(self.input_feat, cuda=self.cuda, device=self.device, 
                critic_fc_units=self.critic_fc_units, generator_fc_units=self.generator_fc_units, 
                critic_fc_layers=self.critic_fc_layers, generator_fc_layers=self.generator_fc_layers, 
                generator_activation=self.generator_activation)

        return WGAN(self.input_feat, cuda=self.cuda, device=self.device,
            critic_fc_units=self.critic_fc_units, generator_fc_units=self.generator_fc_units, 
            critic_fc_layers=self.critic_fc_layers, generator_fc_layers=self.generator_fc_layers, 
            generator_activation=self.generator_activation)

    def pre_train_discriminator(self, dataset):
        raise Exception("NO implementaion")
        
    def save_model(self, path, prod=False):
        models = {}
        for class_index in self.generators:
            gen = self.generators[class_index]
            states = gen.save_model(prod=prod)

            for k in states:
                models[str(class_index)+"_"+k] = states[k]

        torch.save(models, path)

    def load_model(self, path, n_classes=2, prod=False):
        if self.cuda:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path,  map_location='cpu')
        
        for class_index in range(n_classes):
            gan = self.get_model()
            gan.load_model(checkpoint, class_index=class_index, prod=prod)
            self.generators[class_index] = gan
            
            

    def forward(self, x, class_index=None):
        if class_index in self.generators:
            return self.generators[class_index](x)
        
        return

    def _run_train(self, train_dataset, iters, batch_size, loss_cbs, target_transform, replayed_dataset=None):

        iters_left = 1
        cuda = self._is_on_cuda()
        device = self._device()
        for idx, c in enumerate(train_dataset.classes, 0):
            singlelabel_dataset = train_dataset.filter([idx])
            singlelabel_dataset.set_target_tranform(target_transform)
            
            class_index = target_transform(c)
            
            progress = tqdm.tqdm(range(1, iters+1))
            for batch_index in range(1, iters+1):

                iters_left -= 1
                if iters_left==0:
                    data_loader = iter(utils.get_data_loader(singlelabel_dataset, batch_size, cuda=cuda, drop_last=True))
                    iters_left = len(data_loader)

                x, y = next(data_loader)                                    #--> sample training data of current task
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct devi
                scores = None

                if batch_index <= iters:
                    # Train the main model with this batch

                    instance_noise_factor = 0
                    if self.noisy:
                        instance_noise_factor = ((iters - batch_index)*1.0 / iters)

                    loss_dict = self.train_a_batch(x, y, class_index=class_index, noise=instance_noise_factor)

                    for loss_cb in loss_cbs:
                        if loss_cb is not None:
                            loss_cb(progress, batch_index, loss_dict, task=class_index)

            # Close progres-bar(s)
            progress.close()
        
    def train_a_batch(self, x, y, class_index=0, noise=0):

        if class_index not in self.generators:
            gan = self.get_model()
            self.generators[class_index] = gan

        return self.generators[class_index].train_a_batch(x, noise=noise)

    def sample(self, class_index, sample_size):
        z = self.generators[class_index]._noise(sample_size)
        s = self.generators[class_index].generator(z)
        
        return s

    @property
    def name(self):
        return "Generator with multiple GANs"

    def _device(self):
        return self.device

    def _is_on_cuda(self):
        return self.cuda