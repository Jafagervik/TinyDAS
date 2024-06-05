import tinygrad
from tinygrad.nn import Tensor

from abc import abstractmethod

class BaseAE:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, model, f, X: Tensor):
        return f(model(X), X)


