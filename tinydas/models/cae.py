from typing import Dict, Tuple, List, Callable

from tinygrad import TinyJit
from tinygrad.nn import Tensor, Conv2d, ConvTranspose2d
from tinygrad import nn

from tinydas.losses import mse
from tinydas.models.base import BaseAE
from tinydas.utils import minmax

class ConvBlock:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        self.conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def __call__(self, x: Tensor) -> Tensor:
        return self.conv(x).relu()

class DeconvBlock:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=2, output_padding=1):
        self.deconv = ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding)

    def __call__(self, x: Tensor) -> Tensor:
        return self.deconv(x).relu()


class CAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()

        self.sizes = kwargs["mod"]["hidden"] 

        self.encoder: List[Callable[[Tensor], Tensor]] = [
            ConvBlock(self.sizes[0], self.sizes[1], padding=1),
            lambda x: x.max_pool2d(kernel_size=2, stride=2),
            ConvBlock(self.sizes[1], self.sizes[2], padding=1),
            lambda x: x.max_pool2d(kernel_size=2, stride=2),
            ConvBlock(self.sizes[2], self.sizes[3], padding=1),
            lambda x: x.max_pool2d(kernel_size=2, stride=2),
        ]

        self.sizes = self.sizes[::-1]

        self.decoder: List[Callable[[Tensor], Tensor]] = [
            DeconvBlock(8, 8, padding=1, stride=2, output_padding=1),
            DeconvBlock(8, 8, padding=1, stride=2, output_padding=1),  # Upsample
            DeconvBlock(8, 16, padding=1, stride=2, output_padding=2),  # Upsample
            Conv2d(16, 1, 3, padding=1),  # Final layer to match the input channels
            lambda x: x.sigmoid()  # Activation function to constrain the output
        ] 


    @property
    def convolutional(self) -> bool:
        return True

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        y = x.sequential(self.encoder).sequential(self.decoder)
        return (y,)

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        (y,) = self(x)
        loss = mse(y, x)
        return {"loss": loss}


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