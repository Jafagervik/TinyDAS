from typing import List

from dataloader import DataLoader
from dataset import Dataset
from early_stopping import EarlyStopping
from models.ae import AE
from models.base import BaseAE
from tinygrad import GlobalCounters, TinyJit
from tinygrad.nn import Tensor
from tinygrad.nn.optim import Optimizer
from tinygrad.nn.state import load_state_dict, safe_load
from tqdm import trange


class Trainer:
    def __init__(
        self,
        model: BaseAE,
        dataloader: DataLoader,
        optimizer: Optimizer,
        devices: List[str],
        **kwargs,
    ) -> None:
        self.model = model
        if kwargs["load"]:
            self.load_model(kwargs["path"])
        self.dataloader = dataloader
        self.optim = optimizer
        self.devices = devices
        self.best_loss = float("inf")
        self.epochs = kwargs["epochs"] if kwargs["epochs"] else 10
        self.losses = [float(0)] * self.epochs
        self.early_stopping = EarlyStopping(kwargs["patience"], kwargs["min_delta"])

    def load_model(self, path: str) -> None:
        # Example: config/ae/best.safetensors
        full_path = f"{path}/ae/best.safetensors"
        state_dict = safe_load(full_path)
        load_state_dict(self.model, state_dict)
        print(f"Model loaded from {path}")

    @TinyJit
    def _run_epoch(self) -> Tensor:
        running_loss = 0.0
        with Tensor.train():
            # running_loss = Tensor(0.0, requires_grad=False)
            for data, _ in self.dataloader:
                self.optim.zero_grad()
                x = data.reshape(-1, 625 * 2137)
                # x = data.flatten(start_dim=1)
                # if len(self.devices) > 1 and any(self.devices) != "CLANG"

                loss = self.model.criterion(x).backward()

                self.optim.step()

                running_loss += loss.item()
        return Tensor(running_loss)

    def train(self):
        print("Starting training...")
        for epoch in (t := trange(self.epochs)):
            GlobalCounters.reset()
            loss = self._run_epoch()
            self.losses[epoch] = loss.item()
            t.set_description(f"Epoch: {epoch + 1} | loss: {self.losses[epoch]:.2f}")

            self.early_stopping(loss.item())
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    from tinygrad import nn

    model = AE()

    dataset = Dataset(n=10)
    dl = DataLoader(dataset, batch_size=2)
    optim = nn.optim.AdamW(nn.state.get_parameters(model), lr=0.5)
    devices = ["CLANG"]

    trainer = Trainer(
        model, dl, optim, devices, epochs=10, patience=5, min_delta=0.0, load=False
    )

    trainer.train()
