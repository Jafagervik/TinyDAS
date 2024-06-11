from typing import List

from tinygrad import Tensor, nn
from tinygrad.nn.optim import Optimizer

from tinydas.models.ae import AE
from tinydas.models.cnn_ae import CNNAE
from tinydas.models.vae import VAE


def select_optimizer(optimizer: str, parameters: List[Tensor], lr: float) -> Optimizer:
    match optimizer.lower():
        case "adam":
            return nn.optim.Adam(parameters, lr=lr)
        case "adamw":
            return nn.optim.AdamW(parameters, lr=lr)
        case "sgd":
            return nn.optim.SGD(parameters, lr=lr)
        case _:
            return nn.optim.Adam(parameters, lr=lr)


def select_model(model: str, **config):
    match model.lower():
        case "ae":
            return AE(**config)
        case "vae":
            return VAE(**config)
        case "cnnae":
            return CNNAE(**config)
        case _:
            return AE(**config)
