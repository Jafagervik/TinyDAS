import os
from typing import Dict, Tuple, List, Callable

from tinygrad import TinyJit, Tensor
from tinygrad.nn import Conv2d, Linear, ConvTranspose2d, BatchNorm
from tinygrad.nn.state import safe_load, load_state_dict

from tinydas.losses import mse
from tinydas.models.base import BaseAE

class CAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.M = kwargs["mod"]["M"]
        self.N = kwargs["mod"]["N"]
        latent_dim = kwargs["mod"]["latent"]
        hidden_dim = kwargs["mod"]["hidden"]
        self.input_shape = (self.M, self.N)

        self.encoder = []
        for i in range(len(hidden_dim) - 1):
            self.encoder.extend([
                Conv2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, stride=2, padding=1),
                #BatchNorm(hidden_dim[i+1]),
                Tensor.relu
            ])
        self.encoder.extend([
            Conv2d(hidden_dim[-1], hidden_dim[-1], kernel_size=3, stride=2, padding=1),
            Tensor.relu
        ])

        Tensor.no_grad, Tensor.training = True, False
        self.conv_out_shape = self.get_conv_output_shape()
        self.conv_out_size = self.conv_out_shape[1] * self.conv_out_shape[2] * self.conv_out_shape[3]
        Tensor.no_grad, Tensor.training = False, True
        
        self.encoder_linear = Linear(self.conv_out_size, latent_dim)
        self.decoder_linear = Linear(latent_dim, self.conv_out_size)

        hidden_dim = hidden_dim[::-1]

        self.decoder = []
        for i in range(len(hidden_dim) - 1):
            self.decoder.extend([
                ConvTranspose2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                #BatchNorm(hidden_dim[i+1]),
                Tensor.relu
            ])
        self.decoder.append(ConvTranspose2d(hidden_dim[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder.append(Tensor.sigmoid)

    def get_conv_output_shape(self):
        x = Tensor.zeros(1, 1, *self.input_shape)
        x = x.sequential(self.encoder)
        return x.shape

    def encode(self, x):
        x = x.reshape(shape=(-1, 1, *self.input_shape))
        x = x.sequential(self.encoder)
        x = x.reshape(shape=(x.shape[0], -1))
        x = self.encoder_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = x.reshape(shape=(-1, self.conv_out_shape[1], self.conv_out_shape[2], self.conv_out_shape[3]))
        x = x.sequential(self.decoder)
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        y = self.decode(self.encode(x))
        y = y[:, :, :self.M, :self.N]
        return (y,)

    def criterion(self, x: Tensor) -> Tensor:
        (y,) = self(x)
        return mse(x, y)

    @staticmethod
    def loss(out: Tensor, pred: Tensor) -> Tensor: return mse(out, pred)

    def predict(self, x: Tensor) -> Tensor:
        x = x.reshape(1, 1, 625, 2137)
        (out,) = self(x)
        out = out.squeeze()
        return out.realize()

