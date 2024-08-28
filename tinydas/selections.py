from typing import List

from tinydas.lr_schedule import ReduceLROnPlateau
from tinygrad import Tensor, nn
from tinygrad.nn.optim import Optimizer

from tinydas.enums import Opti, LRScheduler
from tinydas.models.ae import AE
from tinydas.models.cae import CAE
from tinydas.models.vae import VAE
from tinydas.models.cvae import CVAE
from tinydas.models.cnnlstm import CNNLSTMAE


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
            print(config["lr"])
            return nn.optim.SGD(parameters, lr=config["lr"], momentum=0.9, nesterov=True, weight_decay=0.00001)
        case _:
            return nn.optim.Adam(
                parameters,
                lr=config["lr"],
                b1=config["b1"],
                b2=config["b2"],
            )

def select_lr_scheduler(lr_sched: LRScheduler, optimizer: Optimizer, **config):
    match lr_sched:
        case LRScheduler.REDUCE:
            return ReduceLROnPlateau(
                optimizer, patience=config["opt"]["patience"], threshold=config["opt"]["threshold"], factor=config["opt"]["factor"]
            )
        case _:
            raise NotImplemented

def select_model(model: str, **config):
    match model.lower():
        case "ae":
            model = AE(**config)
            model.xavier_init()
            return model
        case "vae":
            model = VAE(**config)
            model.vae_init()
            return model
        case "cae":
            model = CAE(**config)
            return model
        case "cvae":
            model = CVAE(**config)
            model.cvae_init()
            return model
        case "cnnlstmae":
            model = CNNLSTMAE(**config)
            return model
        case _:
            raise NotImplementedError
