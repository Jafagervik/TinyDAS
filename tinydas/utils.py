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
    parser.add_argument(
        "--load",
        "-l",
        dest="load",
        action="store_true",
        help="Load a model",
        default=False,
    )

    return parser.parse_args()


def get_gpus(amount: int = 2) -> List[str]:
    return [f"{Device.DEFAULT}:{i}" for i in range(os.getenv("GPUS", amount))]


def get_config(model: str):
    path = os.path.join("configs", model + ".yaml")
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def save_model(model, final: bool = False):
    state_dict = get_state_dict(model)
    model_name = model.__class__.__name__.lower()
    final_or_best = "final.safetensors" if final else "best.safetensors"
    path_to_checkpoints = os.path.join("checkpoints", model_name, final_or_best)
    safe_save(state_dict, path_to_checkpoints)
    print(f"Model saved to {path_to_checkpoints}")


def load_model(model):
    path = os.path.join(
        "checkpoints", model.__class__.__name__.lower(), "best.safetensors"
    )
    state_dict = safe_load(path)
    load_state_dict(model, state_dict)
    print(f"Model loaded from {path}")


def reconstruct(mu: Tensor, logvar: Tensor) -> Tensor:
    std = logvar.exp().sqrt()
    eps = Tensor.randn(mu.shape)
    return mu + eps * std
