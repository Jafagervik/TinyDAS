import os
from typing import Dict, Tuple, List, Callable

from tinydas.convblocks import ConvBlock, DeconvBlock
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
        self.sizes = kwargs["mod"]["hidden"]
        self.input_shape = (self.M, self.N)
        
        self.encoder: List[Callable[[Tensor], Tensor]] = [
            ConvBlock(self.sizes[0], self.sizes[1]),
            ConvBlock(self.sizes[1], self.sizes[2]),
            ConvBlock(self.sizes[2], self.sizes[3])
        ]
        
        self.sizes = self.sizes[::-1]
        self.decoder: List[Callable[[Tensor], Tensor]] = [
            DeconvBlock(self.sizes[0], self.sizes[0], output_padding=1),
            DeconvBlock(self.sizes[0], self.sizes[1], output_padding=1),
            DeconvBlock(self.sizes[1], self.sizes[2], output_padding=2),
            Conv2d(self.sizes[2], 1, 3, padding=1),  # Final layer to match the input channels
            lambda x: x.sigmoid()  
        ]

    def encode(self, x: Tensor) -> Tensor:
        x = x.reshape(shape=(-1, 1, *self.input_shape))
        x = x.sequential(self.encoder)
        return x

    def decode(self, x):
        x = x.sequential(self.decoder)
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        y = self.decode(self.encode(x))
        return (y,)

    def criterion(self, x: Tensor) -> Tensor:
        (y,) = self(x)
        return mse(x, y)

    @staticmethod
    def loss(out: Tensor, pred: Tensor) -> Tensor: return mse(out, pred)

    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        x = x.reshape(1, 1, 625, 2137)
        (out,) = self(x)
        out = out.squeeze()
        return out.realize()