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

        #hidden_layers = hidden_layers[::-1]
        
        # Encoder
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
        return self.decode(z), mu, logvar#.clamp(_min=-20, _max=20)  

    def criterion(self, x: Tensor) -> Tensor:
        x_recon, mu, logvar = self(x)
        # Compute in FP32 for stability
        x_recon = x_recon.float()
        x = x.float()
        mu = mu.float()
        logvar = logvar.float()

        recon_loss = ((x_recon - x) ** 2).sum(axis=1).mean()
        
        # Stabilized KL divergence calculation
        var = logvar.exp().clip(1e-5, 1e5)
        kl_div = 0.5 * (mu.square() + var - 1 - logvar).sum(axis=1).mean()

        total_loss = recon_loss + kl_div * self.kld_weight
        print(f"Recon Loss: {recon_loss.item()}, KL Div: {kl_div.item() * self.kld_weight}, Total Loss: {total_loss.item()}")
        return total_loss


    def reshape(self, x: Tensor) -> Tensor: return x.reshape(-1, self.M, self.N)

    @property
    def convolutional(self) -> bool: return False


"""
class VAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_shape = kwargs["data"]["batch_size"], 1, kwargs["mod"]["M"], kwargs["mod"]["N"]
        self.latent_dim = kwargs["mod"]["latent"]
        self.kld_weight = kwargs["mod"]["kld_weight"]
        
        # Encoder
        self.encoder = self.build_encoder()
        
        # Calculate flatten size
        x = Tensor.zeros(self.input_shape, dtype=dtypes.float16)
        for layer in self.encoder:
            x = layer(x)
        self.flatten_size = x.numel() // x.shape[0]
        
        self.fc_mu = Linear(self.flatten_size, self.latent_dim)
        self.fc_logvar = Linear(self.flatten_size, self.latent_dim)
        
        # Decoder
        self.fc_decoder = Linear(self.latent_dim, self.flatten_size)
        self.decoder = self.build_decoder()

    def build_encoder(self):
        return [
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            lambda x: x.relu(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            lambda x: x.relu(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            lambda x: x.relu(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            lambda x: x.relu(),
        ]

    def build_decoder(self):
        return [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            lambda x: x.relu(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            lambda x: x.relu(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            lambda x: x.relu(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            lambda x: x.sigmoid(),
        ]

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.encoder:
            x = layer(x)
        x = x.flatten(1)
        # Convert to float32 for mu and logvar calculations
        x = x.float()
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp().clip(1e-5, 1e5)
        eps = Tensor.randn(*mu.shape, device=mu.device, dtype=dtypes.float32)
        return (mu + eps * std).half()  # Convert back to float16

    def decode(self, z: Tensor) -> Tensor:
        x = self.fc_decoder(z)
        x = x.reshape((-1, 256, self.input_shape[2] // 16, self.input_shape[3] // 16))
        for layer in self.decoder:
            x = layer(x)
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @property
    def convolutional(self): return True

    def reshape(self, x: Tensor) -> Tensor: return x.rehshape(1,2,3)

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        x_recon, mu, logvar = self(x)
        
        # Compute losses in fp32
        x = x.float()
        x_recon = x_recon.float()
        mu = mu.float()
        logvar = logvar.float()

        recons_loss = ((x - x_recon) ** 2).sum(axis=[1, 2, 3]).mean()
        kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1).mean()
        
        loss = recons_loss + self.kld_weight * kld_loss
        
        # Convert losses back to fp16 for backward pass
        return {
            "loss": loss.half(),
            "rec": recons_loss.half(),
            "kld": kld_loss.half()
        }
"""

"""
class Encoder:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
    ):
        self.net = [
            Linear(input_dim, hidden_dims[0]), 
            lambda x: x.silu()
        ]

        for i in range(len(hidden_dims) - 1):
            self.net.append(Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.net.append(lambda x: x.silu())

        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.sequential(self.net)
        return self.mu(x), self.logvar(x)


class Decoder:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
    ):
        self.net = [
            Linear(latent_dim, hidden_dims[0]),
            lambda x: x.silu()
        ]

        for i in range(len(hidden_dims) - 1):
            self.net.append(Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.net.append(lambda x: x.silu())
            
        self.net.append(Linear(hidden_dims[-1], input_dim))
        self.net.append(lambda x: x.sigmoid())

    
    def custom_sigmoid(self, x: Tensor) -> Tensor:
        return 0.5 * (x / (1 + x.abs())).add(1)

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)
"""


