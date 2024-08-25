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
        self.hidden_dims = kwargs["mod"]["hidden"]  # Should be [16, 32, 64, 128]
        self.kld_weight = kwargs["mod"]["kld_weight"]
        self.input_shape = (self.M, self.N)
        self.beta = kwargs["mod"]["beta"]

        print(f"Input shape: {self.input_shape}")
        print(f"Hidden dims: {self.hidden_dims}")

        # Encoder
        self.encoder = []
        in_channels = 1
        for h_dim in self.hidden_dims:
            self.encoder.extend([
                Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                BatchNorm2d(h_dim),
                Tensor.leakyrelu
            ])
            in_channels = h_dim

        # Add an extra layer to reduce spatial dimensions further
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
        logvar = self.fc_logvar(x).clip(-10, 2)

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

        #recon_loss = mse(x, x_recon)
        #kl_div = kl_divergence(mu, logvar)

        total_loss = recon_loss + kl_div * self.beta #self.kld_weight
        print(f"Recon Loss: {recon_loss.item()}, KL Div: {kl_div.item() * self.kld_weight}, Total Loss: {total_loss.item()}")
        return total_loss

    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        Tensor.no_grad = True
        x = x.reshape(1, 1, self.M, self.N)
        (out,) = self(x)
        out = out.squeeze()
        Tensor.no_grad = False 
        return out.realize()

  