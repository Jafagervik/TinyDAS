from abc import ABC, abstractmethod
import os
from typing import Dict, List, Tuple

from tinydas.utils import model_name
from tinygrad import Tensor, nn, TinyJit, dtypes
from tinygrad.nn.state import safe_load, load_state_dict, safe_save


class BaseAE(ABC):
    """
    Base class for autoencoders with single optimizer.

    # Methods:
    - __init__: Constructor method.
    - __call__: Forward pass.
    - criterion: Loss function.
    - predict: Run a forward pass and return the output.
    - l2reg: Get the l2 regularization
    """

    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]: pass

    @abstractmethod
    def criterion(self, x: Tensor) -> Tensor: pass

    def predict(self, x: Tensor) -> Tensor: pass

    def parameters(self): return nn.state.get_parameters(self)
    def state_dict(self): return nn.state.get_state_dict(self)

    def send_copy(self, devices: List[str]):
        for x in self.state_dict().values(): x.realize().to_(devices)

    def half(self):
        for param in self.state_dict().values():
            param.replace(param.cast(dtypes.half))

    def float(self):
        for param in self.state_dict().values():
            param.replace(param.cast(dtypes.float32))

    def xavier_init(self):
        for name, param in self.state_dict().items():
            if 'weight' in name:
                # Use glorot_uniform for weight initialization
                param.assign(Tensor.glorot_uniform(*param.shape))
            elif 'bias' in name:
                param.assign(Tensor.zeros(*param.shape))

    def he_init(self):
        for name, param in self.state_dict().items():
            if 'weight' in name:
                # He initialization for weights using kaiming_uniform
                param.assign(Tensor.kaiming_uniform(*param.shape, a=0))
            elif 'bias' in name:
                param.assign(Tensor.zeros(*param.shape))

    def vae_init(self):
        for name, param in self.state_dict().items():
            if 'encoder' in name or 'decoder' in name:
                if 'weight' in name:
                    param.assign(Tensor.glorot_uniform(*param.shape))
                elif 'bias' in name:
                    param.assign(Tensor.zeros(*param.shape))
            
            elif 'encoder_mean' in name:
                if 'weight' in name:
                    param.assign(Tensor.uniform(*param.shape, low=-0.01, high=0.01))
                elif 'bias' in name:
                    param.assign(Tensor.zeros(*param.shape))
            
            elif 'encoder_logvar' in name:
                if 'weight' in name:
                    param.assign(Tensor.uniform(*param.shape, low=-0.001, high=0.001))
                elif 'bias' in name:
                    # Initialize log variance biases to a small negative number
                    # This starts the network with low variance
                    param.assign(Tensor.full(shape=param.shape, fill_value=-5))
    
    
    @property
    def dtype(self): return self.parameters()[0].dtype

    @property
    def name(self): return self.__class__.__name__.lower()

    @property
    def l2reg(self): return sum((p**2).sum() for p in self.parameters())

    def save(self, final: bool = False, show: bool = False, debug=False):
        state_dict = self.state_dict()
        if debug:
            for k, v in state_dict.items():
                print(k, v, v.dtype)
                print(v[0, 0].numpy())
                break
        final_or_best = "final" if final else "best"
        path_to_checkpoints = f"/cluster/home/jorgenaf/TinyDAS/checkpoints/{self.name}/{final_or_best}.safetensors"
        safe_save(state_dict, path_to_checkpoints)
        if show:
            print(f"Model saved to {ae}/{final_or_best}.safetensors")


    def load(self): 
        path = os.path.join(
            "./checkpoints",
            f"{self.name}",
            f"best.safetensors",
        )
        load_state_dict(self, safe_load(path))
    
