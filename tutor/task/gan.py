# define dataset and dataloader
from collections.abc import Mapping
from typing import Callable, List, Optional, Sequence, Union
import numpy as np
from mmcv.transforms import to_tensor
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from mmengine.config import Config
from mmengine.dataset import BaseDataset

class MNISTDataset(BaseDataset):
    def __init__(self, 
                 data_root = '',
                 pipeline = ..., 
                 test_mode: bool = False):
        if test_mode:
            mnist_full = MNIST(data_root, train=True, download=True)
            self.mnist_dataset, _ = random_split(mnist_full, [55000, 55000])
        else:
            self.mnist_dataset = MNIST(data_root, train=False, download=True)
            
        super().__init__(data_root=data_root, pipeline=pipeline, test_mode=test_mode)
        
    @staticmethod
    def totensor(img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return to_tensor(img)
    
    def load_data_list(self) -> List[dict]:
        return [
            dict(inputs=self.totensor(np.array(x[0]))) for x in self.mnist_dataset
        ]
        
dataset = MNISTDataset("./data", [])

# build data loader
import os
import torch
from mmengine.runner import Runner

NUM_WORKERS = int(os.cpu_count() / 2)
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

train_loader = dict(
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    persistent_workers = True,
    sampler = dict(type='DefaultSampler', shuffle=True),
    dataset=dataset    
)

train_loader = Runner.build_dataloader(train_loader)

# build networks
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.noise_size = noise_size
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(noise_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), - 1)
        validity = self.model(img_flat)
        
        return validity
    
generator = Generator(100, (1, 28, 28))
discriminator = Discriminator((1, 28, 28))

#  build GAN
from mmengine.model import ImgDataPreprocessor
import torch.nn.functional as F
from mmengine.model import BaseModel

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, List):
        nets = [nets]
    
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
data_preprocessor = ImgDataPreprocessor(mean=([127.5]), std=([127.5]))

class GAN(BaseModel):
    def __init__(self, generator, discriminator, noise_size, data_preprocessor):
        super().__init__(data_preprocessor=data_preprocessor)
        assert generator.noise_size == noise_size
        self.generator = generator
        self.discriminator = discriminator
        self.noise_size = noise_size
    
    def disc_loss(self, disc_pred_fake, disc_pred_real):
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.binary_cross_entropy(disc_pred_fake, 0. * torch.ones_like(disc_pred_fake))
        losses_dict['loss_disc_real'] = F.binary_cross_entropy(disc_pred_real, 1. * torch.ones_like(disc_pred_real))
        
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var
        
    def train_discriminator(self, inputs, optim_wrapper):
        real_imgs = inputs['inputs']
        z = torch.randn((real_imgs.shape[0], self.noise_size)).type_as(real_imgs)
        with torch.no_grad():
            fake_imgs = self.generator(z)
        
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)
        
        parsed_loss, log_vars = self.disc_loss(disc_pred_fake, disc_pred_real)
        optim_wrapper.update_params(parsed_loss)
        return log_vars
    
    def gen_loss(self, disc_pred_fake):
        losses_dict = dict()
        losses_dict['losses_gen'] = F.binary_cross_entropy(disc_pred_fake, 1 * torch.ones_like(disc_pred_fake))
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var
    
    def train_generator(self, inputs, optim_wrapper):
        real_imgs = inputs['inputs']
        z = torch.randn(real_imgs.shape[0], self.noise_size).type_as(real_imgs)
        
        fake_imgs = self.generator(z)
        
        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)
        
        optim_wrapper.update_params(parsed_loss)
        return log_vars
                
    def train_step(self, data, optim_wrapper):
        inputs_dict = self.data_preprocessor(data, True)
        
        # train discriminator
        disc_optimizer_wrapper = optim_wrapper['discriminator']
        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(inputs_dict, disc_optimizer_wrapper)
            
        # train generator
        set_requires_grad(self.discriminator, False)
        gen_optimizer_wrapper = optim_wrapper['generator']
        with gen_optimizer_wrapper.optim_context(self.generator):
            log_vars_gen = self.train_generator(inputs_dict, gen_optimizer_wrapper)
            
        set_requires_grad(self.discriminator, True)
        log_vars.update(log_vars_gen)
        
        return log_vars
    
    def forward(self, batch_inputs, data_samples=None, mode=None):
        return self.generator(batch_inputs)
    
model = GAN(generator, discriminator, 100, data_preprocessor)     

# build optimizer
from mmengine.optim import OptimWrapper, OptimWrapperDict

opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_g_wrapper = OptimWrapper(opt_g)

opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_d_wrapper = OptimWrapper(opt_d)

opt_wrapper_dict = OptimWrapperDict(generator=opt_g_wrapper, 
                                    discriminator=opt_d_wrapper)

# train with runner
train_cfg = dict(by_epoch=True, max_epochs=200)
runner = Runner(
    model=model,
    work_dir='./data',
    train_dataloader=train_loader,
    train_cfg=train_cfg,
    optim_wrapper=opt_wrapper_dict
)

# run
if __name__ == '__main__':
    runner.train()
        


        