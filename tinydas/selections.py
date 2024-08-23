from typing import List

from tinygrad import Tensor, nn
from tinygrad.nn.optim import Optimizer

from tinydas.enums import Opti
from tinydas.models.ae import AE
from tinydas.models.cae import CAE
from tinydas.models.vae import VAE
from tinydas.models.cvae import CVAE


def select_optimizer(optimizer: Opti, parameters: List[Tensor], **config) -> Optimizer:
    match optimizer:
        case Opti.ADAM:
            return nn.optim.Adam(
                parameters,
                lr=config["lr"],
                b1=config["b1"],
                b2=config["b2"],
            )
        case Opti.ADAMW:
            return nn.optim.AdamW(
                parameters,
                lr=config["lr"],
                b1=config["b1"],
                b2=config["b2"],
            )
        case Opti.SGD:
            return nn.optim.SGD(parameters, lr=config["lr"])
        case _:
            return nn.optim.Adam(
                parameters,
                lr=config["lr"],
                b1=config["b1"],
                b2=config["b2"],
            )


def select_model(model: str, **config):
    match model.lower():
        case "ae":
            return AE(**config)
        case "vae":
            return VAE(**config)
        case "cae":
            return CAE(**config)
        case "cvae":
            return CVAE(**config)
        case _:
            raise NotImplementedError
