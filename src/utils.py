import argparse
import os
import random as rnd
from typing import Tuple

import yaml
from tinygrad import Device
from tinygrad.nn import Tensor
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save


def seed_all(seed: int):
    Tensor.manual_seed(seed)
    rnd.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch distributed training")

    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        # default="../configs/vae.yaml",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "ae.yaml")
        ),
    )
    return parser.parse_args()


def get_gpus(amount: int = 2) -> Tuple:
    return tuple([(f"{Device.DEFAULT}:{i}" for i in range(os.getenv("GPUS", amount)))])


def get_config(path: str = "../config/ae.yaml"):
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def save_model(net, path: str):
    safe_save(get_state_dict(net), "model.safetensors")
