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


class Encoder:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        self.net = [
            LinearBlockLayer(input_dim, hidden_dim),
            LinearBlockLayer(hidden_dim, latent_dim),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class Decoder:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        self.net = [
            LinearBlockLayer(latent_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class AE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = Encoder(
            kwargs["M"] * kwargs["N"], kwargs["hidden"], kwargs["latent"]
        )
        self.decoder = Decoder(
            kwargs["M"] * kwargs["N"], kwargs["hidden"], kwargs["latent"]
        )

        self.net = [
            self.encoder,
            self.decoder,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)

    def criterion(self, X: Tensor) -> Tensor:
        return mse(X, self(X))
