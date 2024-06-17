from enum import Enum


class Models(Enum):
    AE = "ae"
    VAE = "vae"
    CNNAE = "cnnae"
    BETAVAE = "betavae"


class Opti(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"

class Normalization(Enum):
    ZSCORE = 0
    MINMAX = 1