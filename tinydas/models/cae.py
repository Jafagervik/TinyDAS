from typing import Dict, Tuple, List, Callable

from tinygrad import TinyJit
from tinygrad.nn import Tensor, Conv2d, ConvTranspose2d, BatchNorm2d
from tinygrad import nn

from tinydas.losses import mse
from tinydas.models.base import BaseAE
from tinydas.utils import minmax

class CAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = [
            Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(16),
            lambda x: x.relu(),
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(32),
            lambda x: x.relu(),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(64),
            lambda x: x.relu(),
        ]
        
        self.decoder = [
            ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(32),
            lambda x: x.relu(),
            ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(16),
            lambda x: x.relu(),
            ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            lambda x: x.sigmoid(),
        ]

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = x.reshape(-1, 1, 625, 2137)
        for layer in self.encoder:
            x = layer(x)
        
        for layer in self.decoder:
            x = layer(x)

        x = x[:, :, :625, :2137]
        
        return (x.squeeze(1), )


    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        (y,) = self(x)
        return mse(y, x)


    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        """
        Input tensor is being processed to fit encoder
        after decoder is done, it is reshaped back
        """
        x = x.reshape(1, 1, 625, 2137)
        (out,) = self(x)

        out = out.reshape(625, 2137)
        return out.realize()