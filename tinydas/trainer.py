from typing import List, Callable

from tinygrad import GlobalCounters, TinyJit
from tinygrad.nn import Tensor
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import colored

from tinydas.dataloader import DataLoader
from tinydas.early_stopping import EarlyStopping
from tinydas.models.base import BaseAE
from tinydas.plots import plot_loss
from tinydas.utils import save_model, printing
from tinydas.timer import Timer


class Trainer:
    def __init__(
        self,
        model: BaseAE,
        dataloader: DataLoader,
        optimizer: Optimizer,
        **kwargs,
    ) -> None:
        self.shape = (kwargs["mod"]["M"], kwargs["mod"]["N"])
        self.model = model
        self.dataloader = dataloader
        self.optim = optimizer
        self.best_loss = float("inf")
        self.epochs = kwargs["epochs"] if kwargs["epochs"] else 10
        self.losses = [float(0)] * self.epochs
        self.early_stopping = EarlyStopping(
            kwargs["es"]["patience"], kwargs["es"]["min_delta"]
        )

    @TinyJit
    def train_step(self, x: Tensor, f: Callable[[Tensor], Tensor]) -> Tensor:
        self.optim.zero_grad()

        x = f(x)

        loss = self.model.criterion(x)["loss"]

        loss.backward()
        self.optim.step()

        return loss.realize()

    def train(self):
        Tensor.training = True

        print(colored(f"Starting training {self.model.__class__.__name__} with {self.epochs} epochs", 'yellow'))
        reshape_fn = lambda x: x.reshape(-1, 1, self.shape[0], self.shape[1]) if self.model.convolutional else x.reshape(-1, self.shape[0] * self.shape[1])
 
        for epoch in range(self.epochs):
            GlobalCounters.reset()

            running_loss = 0.0
            with Timer() as t:
                for data in self.dataloader:
                    running_loss += self.train_step(data, reshape_fn).item()
            self.losses[epoch] = running_loss

            printing(epoch, self.epochs, running_loss, t.interval)

            if running_loss < self.best_loss:
                self.best_loss = running_loss
                save_model(self.model)

            self.early_stopping(running_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                self.losses = self.losses[:epoch + 1]
                print(f"Max loss: {max(self.losses):.4f}, Min loss: {min(self.losses):.4f}")
                save_model(self.model, final=True)
                plot_loss(self.losses, self.model)
                return

        print(f"Max loss: {max(self.losses):.4f}, Min loss: {min(self.losses):.4f}")
        save_model(self.model, final=True)
