import os
from typing import Dict, Tuple, List, Callable

from tinygrad import TinyJit
from tinygrad.nn import Tensor, Conv2d, ConvTranspose2d, Linear, BatchNorm2d

from tinydas.losses import elbo, kl_divergence, mse
from tinydas.models.base import BaseAE
from tinydas.utils import reparameterize

class CVAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()

        self.M = 625
        self.N = 2137
        self.latent_dim = kwargs["mod"]["latent"]
        self.hidden_dims = kwargs["mod"]["hidden"]  
        self.kld_weight = kwargs["mod"]["kld_weight"]
        self.input_shape = (self.M, self.N)
        self.beta = kwargs["mod"]["beta"]

        self.encoder = []
        in_channels = 1
        for h_dim in self.hidden_dims:
            self.encoder.extend([
                Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                BatchNorm2d(h_dim),
                Tensor.leakyrelu
            ])
            in_channels = h_dim

        self.encoder.extend([
            Conv2d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=3, stride=2, padding=1),
            BatchNorm2d(self.hidden_dims[-1]),
            Tensor.leakyrelu
        ])

        Tensor.no_grad, Tensor.training = True, False
        self.conv_out_shape = self.get_conv_output_shape()
        self.conv_out_size = self.conv_out_shape[1] * self.conv_out_shape[2] * self.conv_out_shape[3]
        Tensor.no_grad, Tensor.training = False, True

        self.fc_mu = Linear(self.conv_out_size, self.latent_dim)
        self.fc_logvar = Linear(self.conv_out_size, self.latent_dim)

        self.decoder_input = Linear(self.latent_dim, self.conv_out_size)

        self.decoder = []
        hidden_dims_reversed = self.hidden_dims[::-1]
        
        self.decoder.extend([
            ConvTranspose2d(hidden_dims_reversed[0], hidden_dims_reversed[0],
                            kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(hidden_dims_reversed[0]),
            Tensor.leakyrelu
        ])

        for i in range(len(hidden_dims_reversed) - 1):
            self.decoder.extend([
                ConvTranspose2d(hidden_dims_reversed[i],
                                hidden_dims_reversed[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1),
                BatchNorm2d(hidden_dims_reversed[i + 1]),
                Tensor.leakyrelu
            ])

        # Final layer
        self.final_layers = [
            ConvTranspose2d(hidden_dims_reversed[-1], out_channels=1,
                            kernel_size=3, stride=2, padding=1, output_padding=1),
            Tensor.tanh
        ]

    def get_conv_output_shape(self):
        x = Tensor.zeros(1, 1, *self.input_shape)
        x = x.sequential(self.encoder)
        return x.shape

    def encode(self, x):
        x = x.reshape(shape=(-1, 1, *self.input_shape))
        x = x.sequential(self.encoder)
        x = x.reshape(shape=(x.shape[0], -1))

        mean = self.fc_mu(x)
        logvar = self.fc_logvar(x).clip(-10, 10)

        return mean, logvar

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.reshape(shape=(-1, self.conv_out_shape[1], self.conv_out_shape[2], self.conv_out_shape[3]))
        x = x.sequential(self.decoder)
        x = x.sequential(self.final_layers)
        x = x[:, :, :self.M, :self.N]
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def criterion(self, x: Tensor) -> Tensor:
        x_recon, mu, logvar = self(x)

        # use mse not bce
        recon_loss, kl_div, total_loss = elbo(x, x_recon, mu, logvar, self.beta)

        #print(f"Recon Loss: {recon_loss.item()}, KL Div: {kl_div.item() * self.kld_weight}, Total Loss: {total_loss.item()}")
        return total_loss

    @staticmethod
    def loss(x: Tensor, y: Tensor) -> Tensor: return mse(x, y)

    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        x = x.reshape(1, 1, self.M, self.N)
        out, _, _ = self(x)
        out = out.squeeze()
        return out.realize()

  