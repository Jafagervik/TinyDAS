import argparse
import os
import random as rnd
from typing import List, Tuple

import yaml
from tinygrad import Device
from tinygrad.nn import Tensor
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save


def seed_all(seed: int = 1234) -> None:
    Tensor.manual_seed(seed)
    rnd.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch distributed training")

    parser.add_argument(
        "--type",
        "-t",
        dest="type",
        help="Train or detect mode",
        default="train",
    )
    parser.add_argument(
        "--model",
        "-m",
        dest="model",
        metavar="str",
        help="model to use",
        # default="../configs/vae.yaml",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "configs", "ae.yaml")
        ),
    )

    return parser.parse_args()


def get_gpus(amount: int = 2) -> List[str]:
    return [f"{Device.DEFAULT}:{i}" for i in range(os.getenv("GPUS", amount))]


def get_config(model_type: str = "ae"):
    path = os.path.join(
        os.path.dirname(__file__), "..", "configs", f"{model_type}.yaml"
    )
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def save_model(model, path: str):
    state_dict = get_state_dict(model)
    safe_save(state_dict, path)
    print(f"Model saved to {path}")


def reconstruct(mu: Tensor, logvar: Tensor) -> Tensor:
    std = logvar.exp().sqrt()
    eps = Tensor.randn(mu.shape)
    return mu + eps * std
