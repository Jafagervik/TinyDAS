from typing import Dict, List, Optional, Tuple

from tinygrad.nn import Linear, Tensor
from tinygrad import TinyJit

from tinydas.linearblock import LinearBlockLayer
from tinydas.losses import mse
from tinydas.models.base import BaseAE
from tinydas.utils import minmax

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

    def __call__(self, x: Tensor) -> Tensor:
        x = x.sequential(self.net)
        return self.last(x).sigmoid()


class AE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.M = kwargs["mod"]["M"]
        self.N = kwargs["mod"]["N"]

        hidden_layers = kwargs["mod"]["hidden_layers"]

        self.encoder = Encoder(
            self.M * self.N,
            hidden_layers,
            kwargs["mod"]["latent"],
            kwargs["mod"]["p"],
        )

        hidden_layers = hidden_layers[::-1]

        self.decoder = Decoder(
            self.M * self.N,
            hidden_layers,
            kwargs["mod"]["latent"],
            kwargs["mod"]["p"],
        )

    @property
    def convolutional(self) -> bool:
        return False

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        return (self.decoder(self.encoder(x)),)

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        (x_hat,) = self(x)
        return {"loss": mse(x, x_hat)}


    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        """
        Input tensor is being processed to fit encoder
        after decoder is done, it is reshaped back
        """
        Tensor.no_grad = True
        x = x.reshape(1, 625 * 2137) 
        (out,) = self(x)

        return out.reshape(625, 2137).realize()