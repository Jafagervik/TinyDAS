from typing import Dict, Tuple, List, Callable

from tinygrad import TinyJit
from tinygrad.nn import Tensor, Conv2d, ConvTranspose2d, Linear

from tinydas.losses import elbo, mse
from tinydas.models.base import BaseAE
from tinydas.utils import reparameterize

class CVAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()

        self.M = kwargs["mod"]["M"]
        self.N = kwargs["mod"]["N"]
        self.latent_dim = kwargs["mod"]["latent"]
        self.hidden_dims = kwargs["mod"]["hidden"]
        self.kld_weight = kwargs["mod"]["kld_weight"]
        self.input_shape = (self.M, self.N)

        # Encoder
        self.encoder = []
        in_channels = 1
        for h_dim in self.hidden_dims:
            self.encoder.extend([
                Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                Tensor.relu
            ])
            in_channels = h_dim

        Tensor.no_grad, Tensor.training = True, False
        self.conv_out_shape = self.get_conv_output_shape()
        self.conv_out_size = self.conv_out_shape[1] * self.conv_out_shape[2] * self.conv_out_shape[3]
        Tensor.no_grad, Tensor.training = False, True

        self.fc_mu = Linear(self.conv_out_size, self.latent_dim)
        self.fc_logvar = Linear(self.conv_out_size, self.latent_dim)

        self.decoder_input = Linear(self.latent_dim, self.conv_out_size)

        self.decoder = []
        hidden_dims_reversed = self.hidden_dims[::-1]
        for i in range(len(hidden_dims_reversed) - 1):
            self.decoder.extend([
                ConvTranspose2d(hidden_dims_reversed[i], hidden_dims_reversed[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                Tensor.relu
            ])

        self.decoder.append(ConvTranspose2d(hidden_dims_reversed[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1))

    def get_conv_output_shape(self):
        x = Tensor.zeros(1, 1, *self.input_shape)
        x = x.sequential(self.encoder)
        return x.shape

    def encode(self, x):
        x = x.reshape(shape=(-1, 1, *self.input_shape))
        x = x.sequential(self.encoder)
        x = x.reshape(shape=(x.shape[0], -1))
        mean = self.fc_mu(x)
        logvar = self.fc_logvar(x).clip(-10, 2)  # Clip logvar to a reasonable range
        return mean, logvar

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.reshape(shape=(-1, self.conv_out_shape[1], self.conv_out_shape[2], self.conv_out_shape[3]))
        x = x.sequential(self.decoder)
        x = x[:, :, :self.input_shape[0], :self.input_shape[1]]
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

        print(mu.shape)
        print(logvar.shape)

        recon_loss = mse(x, x_recon)
        kl_div = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(axis=1).mean()

        total_loss = recon_loss + kl_div * self.kld_weight
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