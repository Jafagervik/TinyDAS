from typing import Dict, Tuple

from tinygrad.nn import Tensor

from tinydas.convblocks import ConvBlock, DeconvBlock
from tinydas.losses import mse
from tinydas.models.base import BaseAE


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
        y = x.sequential(self.encoder)
        y = y.sequential(self.decoder)
        y = y.reshape(x.shape[0], self.M, self.N)  # (batch_size, 625, 2137)
        return (y,)

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        (y,) = self(x)
        loss = mse(x, y)
        return {"loss": loss}
