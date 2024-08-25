import argparse
import os
import random as rnd
from sys import getsizeof
from typing import List, Tuple

import h5py
import numpy as np
import yaml
from tinygrad import Device, dtypes, TinyJit, nn
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
    return t.reshape(625, 2137)


def seed_all(seed: int = 1337) -> None:
    Tensor.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)
    print("Seed set to", seed)


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
    devices = [f"{Device.DEFAULT}:{i}" for i in range(amount)]
    for x in devices: Device[x]
    print("Training on", len(devices), "devices")
    return devices


def get_config(model: str):
    path = os.path.join("configs", model + ".yaml")
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

def model_name(model) -> str: return model.__class__.__name__.lower()

def save_model(model, final: bool = False, show: bool = False):
    state_dict = model.state_dict()
    final_or_best = "final.safetensors" if final else "best.safetensors"
    path_to_checkpoints = f"/cluster/home/jorgenaf/TinyDAS/checkpoints/{model.__class__.__name__.lower()}/{final_or_best}"
    safe_save(state_dict, path_to_checkpoints)
    if show:
        print(f"Model saved to {path_to_checkpoints}")

def check_overflow(model):
    for p in model.parameters():
        if p.grad is None: return True
    return False

def load_model(model, name: str = "best", show = True):
    path = os.path.join(
        "./checkpoints",
        model_name(model),
        f"{name}.safetensors",
    )
    load_state_dict(model, safe_load(path))
    if show:
        print(f"Model loaded from {path}")


def reparameterize(mean: Tensor, logvar: Tensor):
    std = (logvar * 0.5).exp() + 1e-5  #.clip(-1e5, 1e5)
    eps = Tensor.randn(mean.shape, device=mean.device)
    return mean + eps * std

def clip_and_grad(optimizer, loss_scaler):
    global_norm = Tensor([0.0], dtype=dtypes.float32, device=optimizer.params[0].device).realize()
    for param in optimizer.params: 
        if param.grad is not None:
            param.grad = param.grad / loss_scaler
            global_norm += param.grad.float().square().sum()

    global_norm = global_norm.sqrt()
    for param in optimizer.params: 
        if param.grad is not None:
            param.grad = (param.grad / Tensor.where(global_norm > 1.0, global_norm, 1.0)).cast(param.grad.dtype)

def new_clip_and_grad(optimizer, loss_scaler, max_norm = 1.00):
    # Compute global norm in float32
    global_norm = Tensor([0.0], dtype=dtypes.float32, device=optimizer.params[0].device) # realize()
    for param in optimizer.params:
        if param.grad is not None:
            # Unscale gradients
            param.grad = param.grad / loss_scaler
            # Accumulate squared sum in float32
            global_norm += param.grad.float().square().sum()
    
    global_norm = global_norm.sqrt()
    
    # Clip gradients
    clip_factor = Tensor.where(global_norm > max_norm, max_norm / global_norm, 1.0)
    for param in optimizer.params:
        if param.grad is not None:
            # Apply clipping and cast back to original dtype
            param.grad = (param.grad * clip_factor).cast(param.dtype)

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
    return (data - data.min()) / (data.max() - data.min())

def zscore(data: Tensor) -> Tensor:
    return data.sub(data.mean()).div(data.std())

def printing(epoch: int, epochs: int, train_loss: float, val_loss: float,
             train_gflops: float, val_gflops: float, lr: float, train_time: float, val_time: float):
    progress = ((epoch + 1) / epochs) * 100
    
    print(colored(f"Epoch {epoch+1}/{epochs}", "green"), end=", ")
    print(colored(f"Train Loss: {train_loss:.9f}", "red"), end=", ")
    print(f"Train GFLOPS: {train_gflops:.2f}", end=", ")
    print(colored(f"Val Loss: {val_loss:.9f}", "blue"), end=", ")
    print(f"Val GFLOPS: {val_gflops:.2f}", end=", ")
    print(f"LR: {lr:.2e}", end=", ")
    print(f"Train Time: {train_time:.2f}s", end=", ")
    print(f"Val Time: {val_time:.2f}s", end=", ")
    print(f"Progress: {progress:.2f}%")

    
def get_anomalous_indices(filepath: str = "anomalous_indices.txt") -> List[str]:
    with open(filepath, 'r') as file:
        return [int(line.strip()) for line in file]

def get_true_anomalies(filepath: str = "anomalous_indices.txt", total_files: int = 600):
    anomalous_indices = get_anomalous_indices(filepath)
    
    true_anomalies = np.zeros(total_files, dtype=bool)

    true_anomalies[anomalous_indices] = True

    return true_anomalies

