from tinygrad import nn, Device
from tinygrad.nn import Tensor
import random as rnd
import argparse
import os
import yaml

def seed_all(seed: int):
    rnd.seed(seed)
    Tensor._seed(int)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch distributed training")

    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        #default="../configs/vae.yaml",
        default=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'config', "ae.yaml"))
    )
    return parser.parse_args()

def get_gpus(amount: int = 2): return [f'{Device.DEFAULT}:{i}' for i in range(os.getenv("GPUS", amount))]


def get_config(path: str = "../config/ae.yaml"):
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1) 