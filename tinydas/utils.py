import argparse
import os
import random as rnd
from sys import getsizeof
from typing import List, Tuple

import h5py
import numpy as np
import yaml
from tinygrad import Device, dtypes, TinyJit
from tinygrad.nn import Tensor
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.helpers import colored


def format_filename(filename: str) -> str:
    #filename = filename or "./infer/20190415_032000.hdf5"
    return filename.split("/")[-1].split(".")[0]
    
def get_size_in_gb(t: Tensor) -> float:
    sz = getsizeof(t) / 1e9
    print(f"Size: {sz:.2f} GB")
    return sz


def custom_flatten(t: Tensor) -> Tensor:
    return t.reshape(1, t.shape[0] * t.shape[1])


def reshape_back(t: Tensor) -> Tensor:
    # TODO: Fix name and hardcoded values
    return t.reshape(625, 2137)


def seed_all(seed: int = 1234) -> None:
    Tensor.manual_seed(seed)
    np.random.seed(seed)
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
    parser.add_argument(
        "--debug",
        "-d",
        dest="debug",
        action="store_true",
        help="Debug",
        default=False,
    )
    parser.add_argument(
        "--gpus",
        "-g",
        dest="gpus",
        metavar="GPUS",
        type=int,
        help="Amount of GPUs",
        default=1,
    )

    return parser.parse_args()


def get_gpus(amount: int = 1) -> List[str]:
    return [f"{Device.DEFAULT}:{i}" for i in range(amount)]


def get_config(model: str):
    path = os.path.join("configs", model + ".yaml")
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def save_model(model, final: bool = False, show: bool = False):
    state_dict = get_state_dict(model)
    final_or_best = "final.safetensors" if final else "best.safetensors"
    path_to_checkpoints = os.path.join(
        "/cluster/home/jorgenaf/TinyDAS/checkpoints",
        # "/home/jaf/prog/ntnu/TinyDAS/checkpoints",
        model.__class__.__name__.lower(),
        final_or_best,
    )
    safe_save(state_dict, path_to_checkpoints)
    if show:
        print(f"Model saved to {path_to_checkpoints}")


def load_model(model):
    path = os.path.join(
        "./checkpoints",
        #"/cluster/home/jorgenaf/TinyDAS/checkpoints",
        #"/home/jaf/prog/ntnu/TinyDAS/checkpoints",
        model.__class__.__name__.lower(),
        "best.safetensors",
    )
    state_dict = safe_load(path)
    load_state_dict(model, state_dict)
    print(f"Model loaded from {path}")


def reparameterize(
    mu: Tensor, 
    logvar: Tensor, 
    dtype = dtypes.float16
) -> Tensor:
    std = (0.5 * logvar).exp()
    eps = Tensor.randn(*std.shape, dtype=dtype, requires_grad=True)
    return (eps * std) + mu


def load_das_file(filename: str):
    """Loads a single das file in to a tuple of the data and the timestamps."""
    with h5py.File(filename, "r") as f:
        data = Tensor(f["raw"][:], dtype=dtypes.float32, requires_grad=False).T
        times = Tensor(f["timestamp"][:], requires_grad=False)
    return data, times

def load_das_file_no_time(filename: str) -> Tensor:
    with h5py.File(filename, "r") as f:
        data = Tensor(f["raw"][:], dtype=dtypes.float32, requires_grad=False).T
    return data
    

def minmax(data: Tensor) -> Tensor:
    return data.sub(data.min()).div(data.max().sub(data.min()))

def zscore(data: Tensor) -> Tensor:
    return data.sub(data.mean()).div(data.std())

def printing(epoch: int, epochs: int, loss: float, dur: float):
    print(colored(f"Epoch {epoch + 1}/{epochs}", "green"), end="\t")
    print(colored(f"Loss: {loss:.7f}", "red"), end="\t")
    print(f"Time: {(dur):.2f}s \t {((epoch+1)/epochs)*100:.2f}%")

    
