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
        

        self.encoder1 = Linear(self.flattened_dim, 512, bias=True)
        self.encoder2 = Linear(512, latent, bias=True)
        
        self.decoder1 = Linear(latent, 512, bias=True)
        self.decoder2 = Linear(512, self.flattened_dim, bias=True)

    def encode(self, x):
        x = x.reshape(shape=(-1, self.flattened_dim))
        x = self.encoder1(x).relu()
        x = self.encoder2(x)
        return x

    def decode(self, x):
        x = self.decoder1(x).relu()
        x = self.decoder2(x)
        #x = x.clip(0, 1)
        x = x.reshape(shape=(-1, self.M, self.N))
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        return (self.decode(self.encode(x)),)

    def criterion(self, x: Tensor) -> Tensor:
        (y, ) = self(x)
        return mse(x,y)
        #return mse(x,y, reduction='none').mean(axis=(1,2)).mean()
        #return ((x-y)**2).mean(axis=(1,2)).mean()

    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        Tensor.no_grad = True
        x = x.reshape(1, 625 * 2137) 
        (out,) = self(x)
        return out.squeeze()