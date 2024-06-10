from tinygrad import Tensor, nn
from tinygrad.device import Device
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save

from tinydas.dataloader import DataLoader
from tinydas.dataset import DataSet
from tinydas.trainer import Trainer
from tinydas.utils import *


class Model:
    def __init__(self):
        self.layers = [
            nn.Linear(100, 10),
            Tensor.gelu,
            nn.Linear(10, 100),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)

    def predict(self, x: Tensor) -> Tensor:
        return self(x)

    def criterion(self, X: Tensor) -> Tensor:
        return self(X).sub(X).square().mean()


def main():
    model = Model()
    path = "model.safetensors"

    # save_model(model, "model.safetensors")

    # and load it back in
    state_dict = safe_load(path)
    load_state_dict(model, state_dict)
    print(f"Model loaded from {path}")


if __name__ == "__main__":
    main()
