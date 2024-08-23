from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

import math
from tinygrad import Tensor, nn, TinyJit, dtypes

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
    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]: pass

    @abstractmethod
    def criterion(self, x: Tensor) -> Tensor: pass

    @TinyJit
    @abstractmethod
    def predict(self, x: Tensor) -> Tensor: pass

    @abstractmethod
    def reshape(self, x: Tensor) -> Tensor: pass

    def parameters(self): return nn.state.get_parameters(self)
    def state_dict(self): return nn.state.get_state_dict(self)

    def send_copy(self, devices: List[str]):
        for x in self.state_dict().values(): x.realize().to_(devices)

    def half(self):
        for param in self.state_dict().values():
            param.replace(param.cast(dtypes.default_float))
        #for l in self.state_dict().values(): l.replace(l.half())

    def float(self):
        for param in self.state_dict().values():
            param.replace(param.cast(dtypes.float32))
        #for l in self.state_dict().values(): l.replace(l.float())

    def xavier_init(self):
        for name, param in self.state_dict().items():
            if 'weight' in name:
                # Use glorot_uniform for weight initialization
                param.assign(Tensor.glorot_uniform(*param.shape))
            elif 'bias' in name:
                # Initialize biases to zero
                param.assign(Tensor.zeros(*param.shape))

    @property
    def dtype(self): 
        """Datatype of model"""
        for x in self.parameters(): return x.dtype