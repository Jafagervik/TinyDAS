from abc import ABC, abstractmethod
from typing import Dict, Tuple

from tinygrad.nn import Tensor
from tinygrad import TinyJit

from tinydas.enums import Models
from tinydas.utils import minmax


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

    @property
    @abstractmethod
    def convolutional(self) -> bool:
        """Helper property to help with how we should reshape before"""
        pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        pass

    @abstractmethod
    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        pass

    
    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        """
        Input tensor is being processed to fit encoder
        after decoder is done, it is reshaped back
        """
        Tensor.no_grad = True
        x = x.reshape(1, 625 * 2137)
        x = minmax(x)
        print(f"Model: {self.__class__.__name__.lower()}")
        match self.__class__.__name__.lower():
            case Models.VAE | Models.BETAVAE:
                (out, _, _) = self(x)
            case Models.AE | Models.CNNAE:
                (out,) = self(x)
            case _:
                return Tensor.empty().realize()

        out = out.reshape(625, 2137)
        Tensor.no_grad = False
        return out.realize()
