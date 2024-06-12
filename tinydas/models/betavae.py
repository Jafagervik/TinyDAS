"""Beta variational autoencoder model."""

from tinydas.losses import mse
from tinygrad import Tensor
from tinygrad.nn import Conv2D, BatchNorm2D, ConvTranspose2D, Linear
from typing import List, Tuple

from tinydas.utils import reparameterize


class BetaVAE:
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = Tensor([max_capacity], requires_grad=False)
        self.C_stop_iter = Capacity_max_iter

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder_layers = []
        self.in_channels = in_channels

        # Build Encoder
        for h_dim in hidden_dims:
            self.encoder_layers.append(Conv2D(self.in_channels, h_dim, kernel_size=3, stride=2, padding=1))
            self.encoder_layers.append(BatchNorm2D(h_dim))
            self.encoder_layers.append(LeakyReLU())
            self.in_channels = h_dim

        self.fc_mu = Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        self.decoder_layers = []
        self.decoder_input = Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            self.decoder_layers.append(
                ConvTranspose2D(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            self.decoder_layers.append(BatchNorm2D(hidden_dims[i + 1]))
            self.decoder_layers.append(LeakyReLU())

        self.final_layer = [
            ConvTranspose2D(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2D(hidden_dims[-1]),
            LeakyReLU(),
            Conv2D(hidden_dims[-1], 3, kernel_size=3, padding=1),
            Tanh()
        ]

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [Tensor(mu), Tensor(log_var)]

    def decode(self, z: Tensor) -> Tensor:
        z = self.decoder_input(z)
        z = z.reshape(z.shape[0], 512, 2, 2)
        for layer in self.decoder_layers:
            z = layer(z)
        for layer in self.final_layer:
            z = layer(z)
        return z

    def __call__(self, x: Tensor) -> Tensor:
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        return Tensor([self.decode(z), x, mu, log_var])

    def loss_function(self, *args, **kwargs) -> Tensor:
        self.num_iter += 1
        recons = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = mse(recons, x)

        kld_loss = (-0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(axis=1)).mean()

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(x.device)
            C = min(self.C_max / self.C_stop_iter * self.num_iter, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return Tensor([loss, recons_loss, kld_loss])

    def sample(self, num_samples: int, current_device: int) -> Tensor:
        z = Tensor.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)
