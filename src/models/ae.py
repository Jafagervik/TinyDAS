from tinygrad.nn import Tensor
from .base import BaseAE
from ..loss import mse

class AE(BaseAE):
    def __init__(self):
        super().__init__()
        self.net = [

        ]

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def loss_function(self, model, f, X: Tensor):
        return super().loss_function(model, f, X)