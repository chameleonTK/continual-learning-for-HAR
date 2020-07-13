from functools import reduce
import torch
from torch import nn, autograd
from torch.autograd import Variable
import utils
from replayer import Replayer
from model_cgan import CGAN
from model_cwgan import CWGAN
# from model_gan import GAN

import numpy as np
import tqdm

EPSILON = 1e-16
class GeneratorSingleGAN(Replayer):
    def __init__(self, input_feat, classes, cuda=False, device="cpu",
            model="cwgan", z_size=20,
            critic_fc_layers=3, critic_fc_units=100, critic_lr=1e-03, 
            generator_fc_layers=3, generator_fc_units=100, generator_lr=1e-03,
            generator_activation="relu"):
        super().__init__()
        self.label = "Generator with one single GAN"
        self.model = model
        self.cuda = cuda
        self.device = device
        
        self.input_feat = input_feat
        self.classes = classes

        self.critic_fc_units = critic_fc_units
        self.critic_fc_layers = critic_fc_layers

        self.generator_fc_units = generator_fc_units
        self.generator_fc_layers = generator_fc_layers

        self.generator_activation = generator_activation
        
        self.generator = self.get_model()
        self.noisy = False


    def get_model(self):
        if self.model == "cgan":
            return CGAN(self.input_feat, self.classes, 
            critic_fc_units=self.critic_fc_units, generator_fc_units=self.generator_fc_units, 
            critic_fc_layers=self.critic_fc_layers, generator_fc_layers=self.generator_fc_layers, 
            generator_activation=self.generator_activation)

        return CWGAN(self.input_feat, self.classes, 
            critic_fc_units=self.critic_fc_units, generator_fc_units=self.generator_fc_units, 
            critic_fc_layers=self.critic_fc_layers, generator_fc_layers=self.generator_fc_layers, 
            generator_activation=self.generator_activation)

    def save_model(self, path, prod=False):
        models = self.generator.save_model(prod=prod)
        torch.save(models, path)

    def load_model(self, path, n_classes=2, prod=False):
        if self.cuda:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path,  map_location='cpu')

        self.generator.load_model(checkpoint, class_index=None, prod=prod)

    def forward(self, x, class_index=None):
        raise Exception("NO implementaion")

    def pre_train_discriminator(self, dataset):
        def target_transform(x):
            return 0

        self._run_train(dataset, 1000, 5, [], target_transform)

        # return self.generator.pre_train_discriminator(dataset)
    

    def _run_train(self, train_dataset, iters, batch_size, loss_cbs, target_transform, replayed_dataset=None, loss_tracking=None):


        # Reset CGAN
        self.generator = self.get_model()

        iters_left = 1
        replay_iters_left = 1
        cuda = self._is_on_cuda()
        device = self._device()

        c = train_dataset.classes[0]
        class_index = target_transform(c)

        progress = tqdm.tqdm(range(1, iters+1))
        for batch_index in range(1, iters+1):

            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(train_dataset, min(batch_size, len(train_dataset)), cuda=cuda, drop_last=True))
                iters_left = len(data_loader)

            if replayed_dataset is not None:
                replay_iters_left -=1
                if replay_iters_left==0:
                    replayed_data_loader = iter(utils.get_data_loader(replayed_dataset, min(batch_size, len(replayed_dataset)), cuda=cuda, drop_last=True))
                    replay_iters_left = len(replayed_data_loader)

            x, y = next(data_loader)                                    #--> sample training data of current task
            x, y = x.to(device), y.to(device)                           #--> transfer them to correct devi
            scores = None

            instance_noise_factor = 0
            if self.noisy:
                instance_noise_factor = ((iters - batch_index)*1.0 / iters)

            # Train the main model with this batch
            if replayed_dataset is not None:
                try: 
                    x_, y_ = next(replayed_data_loader)                               
                    x_, y_ = x_.to(device), y_.to(device) 
                    loss_dict = self.train_a_batch(x_, y_, noise=instance_noise_factor)
                except StopIteration:
                    continue

            loss_dict = self.train_a_batch(x, y, noise=instance_noise_factor)

            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, task=class_index)
            
            if class_index not in loss_tracking["gan_loss"]:
                loss_tracking["gan_loss"][class_index] = []

            loss_tracking["gan_loss"][class_index].append(loss_dict)

        # Close progres-bar(s)
        progress.close()
        return self.generator

    def train_a_batch(self, x, y, class_index=0, noise=0):
        return self.generator.train_a_batch(x, y, noise=noise)

    def sample(self, class_index, sample_size):

        mode = self.generator.generator.training
        self.generator.generator.eval()

        z = self.generator._noise(sample_size)
        y = torch.LongTensor(list(np.full((sample_size, ), int(class_index))))
        s = self.generator.generator(z, y)

        self.generator.generator.train(mode=mode)
        
        return s

    @property
    def name(self):
        return "Generator with one single GAN"

    def _device(self):
        return self.device

    def _is_on_cuda(self):
        return self.cuda