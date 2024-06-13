from abc import ABC, abstractmethod

from tinygrad.nn import Tensor

from typing import Tuple, Dict


class BaseAE(ABC):
    """
    Base class for autoencoders with single optimizer.

    # Methods:
    - __init__: Constructor method.
    - __call__: Forward pass.
    - criterion: Loss function.
    - predict: Run a forward pass and return the output.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        pass

    @abstractmethod
    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        pass

    def predict(self, x: Tensor) -> Tensor:
        """
        Input tensor is being processed to fit encoder
        after decoder is done, it is reshaped back
        """
        Tensor.no_grad = True
        x = x.reshape(-1, 625 * 2137)
        out = self(x)
        out = out.reshape(625, 2137)
        Tensor.no_grad = False 
        return out.realize()
