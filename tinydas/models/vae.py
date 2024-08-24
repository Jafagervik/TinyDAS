from math import prod
from typing import Dict, List, Tuple

import numpy as np
from tinydas.kl import AdaptiveKLWeight
from tinygrad import nn, TinyJit, dtypes
from tinygrad.nn import Tensor, Linear

from tinydas.linearblock import LinearBlockLayer
from tinydas.losses import cross_entropy, kl_divergence, mse, elbo
from tinydas.models.base import BaseAE
from tinydas.utils import reparameterize, minmax

class VAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()

        self.M = kwargs["mod"]["M"]
        self.N = kwargs["mod"]["N"]

        self.hidden= kwargs["mod"]["hidden"]
        self.input_shape = (self.M, self.N)
        self.flattened_dim = self.M * self.N
        self.latent_dim = kwargs["mod"]["latent"]
        self.kld_weight = kwargs["mod"]["kld_weight"]

        self.encoder1 = Linear(self.flattened_dim, 512, bias=True)
        self.encoder_mean = Linear(512, self.latent_dim, bias=True)
        self.encoder_logvar = Linear(512, self.latent_dim, bias=True)
        
        # Decoder
        self.decoder1 = Linear(self.latent_dim, 512, bias=True)
        self.decoder2 = Linear(512, self.flattened_dim, bias=True)

    def encode(self, x):
        x = x.reshape(shape=(-1, self.flattened_dim))
        h = self.encoder1(x).relu()
        mean = self.encoder_mean(h)
        logvar = self.encoder_logvar(h).clip(-10, 2)  # Clip logvar to a reasonable range
        return mean, logvar

    def decode(self, z):
        h = self.decoder1(z).relu()
        x = self.decoder2(h)
        x = x.reshape(shape=(-1, *self.input_shape))
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def criterion(self, x: Tensor) -> Tensor:
        x_recon, mu, logvar = self(x)
        x_recon = x_recon.float()
        x = x.float()
        mu = mu.float()
        logvar = logvar.float()

        recon_loss = mse(x, x_recon)
        kl_div = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(axis=1).mean()

        total_loss = recon_loss + kl_div * self.kld_weight
        #print(f"Recon Loss: {recon_loss.item()}, KL Div: {kl_div.item() * self.kld_weight}, Total Loss: {total_loss.item()}")
        return total_loss
