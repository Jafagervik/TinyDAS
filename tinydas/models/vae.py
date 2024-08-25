from math import prod
import os
from typing import Dict, List, Tuple

import numpy as np
from tinydas.kl import AdaptiveKLWeight, KLAnnealer
from tinygrad import nn, TinyJit, dtypes
from tinygrad.nn import Tensor, Linear
from tinygrad.nn.state import safe_load, load_state_dict

from tinydas.linearblock import LinearBlockLayer
from tinydas.losses import BCE, cross_entropy, kl_divergence, mse, elbo
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
        self.latent = kwargs["mod"]["latent"]
        self.beta = kwargs["opt"]["beta"]
        #self.kld_weight = kwargs["mod"]["kld_weight"]

        self.kld_weight = KLAnnealer(start=0, stop=1, n_steps=1000) 

        self.encoder = []
        in_features = self.flattened_dim
        for h in self.hidden:
            self.encoder.append(Linear(in_features, h))
            self.encoder.append(Tensor.relu)
            in_features = h

        self.encoder_mean = Linear(in_features, self.latent, bias=True)
        self.encoder_logvar = Linear(in_features, self.latent, bias=True)

        self.decoder = []
        in_features = self.latent
        for h in reversed(self.hidden):
            self.decoder.append(Linear(in_features, h))
            self.decoder.append(Tensor.relu)
            in_features = h
        self.decoder.append(Linear(in_features, self.flattened_dim))

    def encode(self, x: Tensor):
        x = x.reshape(shape=(-1, self.flattened_dim))
        x = x.sequential(self.encoder)
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x) #.clip(-10, 2)  # Clip logvar to a reasonable range
        return mean, logvar

    def decode(self, z: Tensor):
        x = z.sequential(self.decoder)
        x = x.reshape(shape=(-1, *self.input_shape))
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def criterion(self, x: Tensor) -> Tensor:
        x_recon, mu, logvar = self(x)
        #x_recon, x, mu, logvar = x_recon.float()

        recon_loss, kl_div, total_loss = elbo(x, x_recon, mu, logvar, self.beta)

        print(f"Recon Loss: {recon_loss.item():.8f}, KL Div: {kl_div.item() * self.beta:.3e}, Total Loss: {total_loss.item():.8f}")
        return total_loss


    def predict(self, x: Tensor) -> Tensor:
        x = x.reshape(1, 625, 2137)
        out, _, _ = self(x)
        out = out.squeeze()
        return out.realize()


#recon_loss = mse(x, x_recon)
#kl_div = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum()