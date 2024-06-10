from typing import Any, Optional

from tinygrad import dtypes, nn
from tinygrad.nn import Tensor

from tinydas.losses import mse
from tinydas.models.base import BaseAE

from ..losses import mae, mse
from .base import BaseAE


class CNNAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.M = kwargs["M"] or 625
        self.N = kwargs["N"] or 2137
        self.inp = self.M * self.N
        self.hidden = 512  # kwargs["hidden"] or 128
        self.latent = 256  # kwargs["latent"] or 64
        self.f = mse if kwargs["loss"] == "mse" else mae

        self.net = [
            nn.Linear(self.hidden, self.inp),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)

    def criterion(self, X: Tensor) -> Tensor:
        return self.f(X, self(X))
