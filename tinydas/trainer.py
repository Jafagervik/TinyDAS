from typing import List

from tinygrad import GlobalCounters, TinyJit
from tinygrad.nn import Tensor
from tinygrad.nn.optim import Optimizer
from tqdm import trange

from tinydas.dataloader import DataLoader
from tinydas.early_stopping import EarlyStopping
from tinydas.models.base import BaseAE
from tinydas.utils import save_model


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
        self.dataloader = dataloader
        self.optim = optimizer
        self.devices = devices
        self.best_loss = float("inf")
        self.epochs = kwargs["epochs"] if kwargs["epochs"] else 10
        self.losses = [float(0)] * self.epochs
        self.early_stopping = EarlyStopping(kwargs["patience"], kwargs["min_delta"])

    @TinyJit
    def _run_epoch(self) -> Tensor:
        running_loss = 0.0
        with Tensor.train():
            for x in self.dataloader:
                if len(self.devices) > 1:
                    x.shard_(self.devices, axis=0)
                x = x.reshape(-1, 625 * 2137)

                loss = self.model.criterion(x)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                running_loss += loss.item()
        return Tensor(running_loss)

    def train(self):
        print("Starting training...")
        for epoch in (t := trange(self.epochs)):
            GlobalCounters.reset()
            loss = self._run_epoch()
            li = loss.item()
            self.losses[epoch] = li
            msg = f"Epoch: {epoch + 1} | Loss: {li:.2f}"
            if li < self.best_loss:
                self.best_loss = li
                self.early_stopping.best_loss = self.best_loss
                save_model(self.model)
                msg  = f"New best loss: {li:.2f}"
                
            t.set_description(msg)

            self.early_stopping(li)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                save_model(self.model, final=True)
                break
        save_model(self.model, final=True)
