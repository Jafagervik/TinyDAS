from typing import Optional

from losses import mse
from tinygrad import dtypes, nn
from tinygrad.nn import Tensor

from .base import BaseAE


class LL:
    def __init__(self, i: int, o: int, do: Optional[float] = None) -> None:
        self.net = [
            nn.Linear(i, o),
            Tensor.relu,
        ]
        if do:
            self.net.append(Tensor.dropout)

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class AE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.M = 625
        self.N = 2137
        self.inp = self.M * self.N
        self.hidden = 64
        self.latent = 8

        self.net = [
            LL(self.inp, self.hidden, do=0.2),
            LL(self.hidden, self.latent),
            LL(self.latent, self.hidden),
            nn.Linear(self.hidden, self.inp),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        # x = x.reshape(-1, self.inp)
        return x.sequential(self.net)

    def criterion(self, X: Tensor) -> Tensor:
        return mse(X, self(X))
