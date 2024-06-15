from typing import Dict, List, Tuple

from tinygrad.nn import Tensor

from tinydas.linearblock import LinearBlockLayer
from tinydas.losses import mse
from tinydas.models.base import BaseAE


class Encoder:
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        self.net = [
            LinearBlockLayer(input_dim, hidden_dims[0]),
        ]
        for i in range(len(hidden_dims) - 1):
            self.net.append(LinearBlockLayer(hidden_dims[i], hidden_dims[i + 1]))
        self.net.append(LinearBlockLayer(hidden_dims[-1], latent_dim))

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class Decoder:
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        self.net = [
            LinearBlockLayer(latent_dim, hidden_dims[0]),
        ]
        for i in range(len(hidden_dims) - 1):
            self.net.append(LinearBlockLayer(hidden_dims[i], hidden_dims[i + 1]))
        self.net.append(LinearBlockLayer(hidden_dims[-1], input_dim))

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class AE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()

        hidden_layers = kwargs["mod"]["hidden_layers"]

        self.encoder = Encoder(
            kwargs["mod"]["M"] * kwargs["mod"]["N"],
            hidden_layers,
            kwargs["mod"]["latent"],
        )

        hidden_layers = hidden_layers[::-1]

        self.decoder = Decoder(
            kwargs["mod"]["M"] * kwargs["mod"]["N"],
            hidden_layers,
            kwargs["mod"]["latent"],
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        return (self.decoder(self.encoder(x)),)

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        (x_hat,) = self(x)
        return {"loss": mse(x, x_hat)}
