from tinygrad.nn import Tensor
from tinygrad import nn
from ..loss import mse


class AE(BaseAE):
    def __init__(self, **kwargs):
        self.M = kwargs["M"]
        self.N = kwargs["N"]
        self.inp = self.M * self.N
        self.hidden = kwargs["hidden"]
        self.latent = kwargs["latent"]

        self.net = [
            nn.Linear(self.inp, 2048), nn.gelu, Tensor.dropout,
            nn.Linear(2048, 1024), nn.gelu, Tensor.dropout,
            nn.Linear(1024, 256), nn.gelu, Tensor.dropout,
            nn.Linear(256, 128), nn.gelu, Tensor.dropout,

            nn.Linear(128, 256), nn.gelu, Tensor.dropout,
            nn.Linear(256, 1024), nn.gelu, Tensor.dropout,
            nn.Linear(1024, 2048), nn.gelu, Tensor.dropout,
            nn.Linear(2048, self.inp), nn.sigmoid
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return nn.sequential(self.net(x))

    def loss_function(self, model, X: Tensor):
        return mse(model, X)