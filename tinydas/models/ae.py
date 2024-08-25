from typing import Dict, List, Optional, Tuple

from tinygrad.nn import Linear, Tensor
from tinygrad import TinyJit

from tinydas.losses import mse
from tinydas.models.base import BaseAE

class AE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.M = kwargs["mod"]["M"]
        self.N = kwargs["mod"]["N"]
        hidden_layers = kwargs["mod"]["hidden"]
        latent  = kwargs["mod"]["latent"]
        input_shape = (self.M, self.N)

        self.input_shape = input_shape
        self.flattened_dim = input_shape[0] * input_shape[1]

        self.encoder = []
        in_features = self.flattened_dim
        for h in hidden_layers:
            self.encoder.append(Linear(in_features, h))
            self.encoder.append(Tensor.relu)
            in_features = h
        self.encoder.append(Linear(in_features, latent))

        self.decoder = []
        in_features = latent
        for h in reversed(hidden_layers):
            self.decoder.append(Linear(in_features, h))
            self.decoder.append(Tensor.relu)
            in_features = h
        self.decoder.append(Linear(in_features, self.flattened_dim))
            
    def encode(self, x):
        x = x.reshape(shape=(-1, self.flattened_dim))
        x = x.sequential(self.encoder)
        return x

    def decode(self, x):
        x = x.sequential(self.decoder)
        x = x.reshape(shape=(-1, self.M, self.N))
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        return (self.decode(self.encode(x)),)

    def criterion(self, x: Tensor) -> Tensor:
        (y, ) = self(x)
        return mse(x,y)

    @staticmethod
    def loss(out: Tensor, pred: Tensor) -> Tensor: return mse(out, pred)

    def predict(self, x: Tensor) -> Tensor:
        x = x.reshape(1, 625 * 2137) 
        (out,) = self(x)
        res = out.squeeze()
        return res