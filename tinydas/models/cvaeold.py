from typing import Dict, Tuple, List, Callable

from tinygrad import TinyJit
from tinygrad.nn import Tensor, Conv2d, ConvTranspose2d, Linear
from tinygrad import nn
from itertools import chain

from tinydas.losses import elbo, mse
from tinydas.models.base import BaseAE
from tinydas.utils import reparameterize


class CVAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_shape = (kwargs["data"]["batch_size"], 1, kwargs["mod"]["M"], kwargs["mod"]["N"])
        self.latent_dim = kwargs["mod"]["latent"]
        self.kld_weight = kwargs["mod"]["kld_weight"]
        
        hidden_dims = kwargs["mod"]["hidden"] #or [32, 64, 128, 256]
        
        # Encoder
        self.encoder = []
        in_channels = self.input_shape[1]
        for h_dim in hidden_dims:
            self.encoder.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                Tensor.relu
            ])
            in_channels = h_dim
        
        Tensor.training, Tensor.no_grad = False, True
        x = Tensor.zeros(self.input_shape)
        x = x.sequential(self.encoder)
        self.flatten_size = x.numel() // x.shape[0]
        self.encoder_output_shape = x.shape[1:]
        print(f"Flatten size: {self.flatten_size}")
        print(f"Encoder output shape: {self.encoder_output_shape}")
        Tensor.training, Tensor.no_grad = True, False

        self.fc_mu = Linear(self.flatten_size, self.latent_dim)
        self.fc_logvar = Linear(self.flatten_size, self.latent_dim)
        
        self.fc_decoder = Linear(self.latent_dim, self.flatten_size)
        
        self.decoder = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            self.decoder.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]), 
                Tensor.relu
            ])
        
        self.final_layer = [
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]), 
            Tensor.relu,
            nn.Conv2d(hidden_dims[-1], self.input_shape[1], kernel_size=3, padding=1),
            Tensor.sigmoid
        ]

    @property
    def convolutional(self) -> bool:
        return True

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.sequential(self.encoder)
        x = x.flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z: Tensor) -> Tensor:
        x = self.fc_decoder(z)
        x = x.reshape((-1,) + self.encoder_output_shape)
        x = x.sequential(self.decoder)
        x = x.sequential(self.final_layer)
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        x_recon = self.decode(z)
        # Crop the output to match the input size
        x_recon = x_recon[:, :, :self.input_shape[2], :self.input_shape[3]]
        return x_recon, mu, logvar

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        x_hat, mu, logvar = self(x)
        tot_loss = elbo(x, x_hat, mu, logvar)
        return {"loss": tot_loss}

    def reshape(self, x: Tensor) -> Tensor: return x.reshape(-1, 1, *x.shape)

    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        """
        Input tensor is being processed to fit encoder
        after decoder is done, it is reshaped back
        """
        Tensor.no_grad = True
        x = x.reshape(1, 1, 625, 2137)
        (out,) = self(x)

        out = out.reshape(625, 2137)
        return out.realize()