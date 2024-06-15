from typing import Dict, List, Optional, Tuple

from tinygrad.nn import Linear, Tensor

from tinydas.linearblock import LinearBlockLayer
from tinydas.losses import mse
from tinydas.models.base import BaseAE


class Encoder:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        do: float,
    ):
        self.net = [
            LinearBlockLayer(input_dim, hidden_dims[0], do),
        ]
        for i in range(len(hidden_dims) - 1):
            self.net.append(LinearBlockLayer(hidden_dims[i], hidden_dims[i + 1], do))
        self.net.append(LinearBlockLayer(hidden_dims[-1], latent_dim, do))

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class Decoder:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        do: float,
    ):
        self.net = [
            LinearBlockLayer(latent_dim, hidden_dims[0], do),
        ]
        for i in range(len(hidden_dims) - 1):
            self.net.append(LinearBlockLayer(hidden_dims[i], hidden_dims[i + 1]))
        self.last = Linear(hidden_dims[-1], input_dim)
        # self.net.append(Tensor.sigmoid)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.sequential(self.net)
        return self.last(x).sigmoid()


class AE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()

        hidden_layers = kwargs["mod"]["hidden_layers"]

        self.encoder = Encoder(
            kwargs["mod"]["M"] * kwargs["mod"]["N"],
            hidden_layers,
            kwargs["mod"]["latent"],
            kwargs["mod"]["p"],
        )

        hidden_layers = hidden_layers[::-1]

        self.decoder = Decoder(
            kwargs["mod"]["M"] * kwargs["mod"]["N"],
            hidden_layers,
            kwargs["mod"]["latent"],
            kwargs["mod"]["p"],
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        return (self.decoder(self.encoder(x)),)

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        (x_hat,) = self(x)
        return {"loss": mse(x, x_hat)}
