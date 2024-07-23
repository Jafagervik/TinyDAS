from enum import Enum


class Models(Enum):
    """Models to use"""
    AE = "ae"
    VAE = "vae"
    CNNAE = "cnnae"
    CAE = "cae"
    BETAVAE = "betavae"


class Opti(Enum):
    """Optimizer to use"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"

class Normalization(Enum):
    """Normalization method"""
    ZSCORE = 0
    MINMAX = 1