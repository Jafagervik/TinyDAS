from typing import Dict, Tuple

from tinygrad import dtypes, nn
from tinygrad.nn import Tensor

from tinydas.convblocks import ConvBlock, DeconvBlock
from tinydas.losses import mse
from tinydas.models.base import BaseAE


class CNNAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.M = kwargs["mod"]["M"] or 625
        self.N = kwargs["mod"]["N"] or 2137
        self.inp = self.M * self.N

        sizes = [1, 16, 32, 64, 128]

        self.encoder = [ConvBlock(i, i + 1) for i in range(len(sizes) - 1)]

        sizes = sizes[::-1]

        self.decoder = [DeconvBlock(i, i + 1) for i in range(len(sizes) - 1)]

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        # Reshape the input tensor to include a channel dimension
        y = x.reshape(-1, 1, self.M, self.N)  # (batch_size, 1, 625, 2137)
        y = y.sequential(self.encoder)
        y = y.sequential(self.decoder)
        # Reshape the output tensor to the original input shape without the channel dimension
        y = y.reshape(x.shape[0], x.shape[2], x.shape[3])  # (batch_size, 625, 2137)
        return x, y

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        x, y = self(x)
        loss = mse(x, y)
        return {"loss": loss}
