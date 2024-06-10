from typing import Optional

from tinygrad import dtypes, nn
from tinygrad.nn import Tensor

from ..losses import mse
from .base import BaseAE


class LinearBlockLayer:
    def __init__(self, i: int, o: int, do: Optional[float] = None):
        self.net = [
            nn.Linear(i, o),
            # Tensor.batchnorm,
            Tensor.relu,
        ]
        if do:
            self.net.append(Tensor.dropout)

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class AE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.M = kwargs["M"] or 625
        self.N = kwargs["N"] or 2137
        self.inp = self.M * self.N
        self.hidden = 512  # kwargs["hidden"] or 128
        self.latent = 256  # kwargs["latent"] or 64

        self.net = [
            LinearBlockLayer(self.inp, self.hidden, do=0.2),
            LinearBlockLayer(self.hidden, self.latent),
            LinearBlockLayer(self.latent, self.hidden),
            nn.Linear(self.hidden, self.inp),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)

    def criterion(self, X: Tensor) -> Tensor:
        return mse(X, self(X))
