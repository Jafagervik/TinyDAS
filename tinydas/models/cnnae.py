from typing import Dict, Tuple

from tinygrad import TinyJit
from tinygrad.nn import Tensor

from tinydas.convblocks import ConvBlock, DeconvBlock
from tinydas.losses import mse
from tinydas.models.base import BaseAE
from tinydas.utils import minmax


class CNNAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.M = kwargs["mod"]["M"] 
        self.N = kwargs["mod"]["N"] 
        self.inp = self.M * self.N

        self.sizes = kwargs["mod"]["hidden"] 

        self.encoder = [
            ConvBlock(self.sizes[i], self.sizes[i + 1], stride=1, padding=1)
            for i in range(len(self.sizes) - 1)
        ]

        self.sizes = self.sizes[::-1]

        self.decoder = [
            DeconvBlock(
                self.sizes[i], self.sizes[i + 1], stride=1, padding=1, output_padding=0
            )
            for i in range(len(self.sizes) - 1)
        ]

    @property
    def convolutional(self) -> bool:
        return True

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        y = x.sequential(self.encoder).sequential(self.decoder)
        y = y.reshape(x.shape[0], self.M, self.N)  # (batch_size, 625, 2137)
        return (y,)

    def criterion(self, x: Tensor, loss_type: str = "bce") -> Dict[str, Tensor]:
        (y,) = self(x)
        loss = y.binary_crossentropy(x) if loss_type == "bce" else mse(x, y)
        return {"loss": loss}


    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        """
        Input tensor is being processed to fit encoder
        after decoder is done, it is reshaped back
        """
        Tensor.no_grad = True
        x = x.reshape(1, 1, 625, 2137)
        x = minmax(x)
        (out,) = self(x)

        out = out.reshape(625, 2137)
        return out.realize()