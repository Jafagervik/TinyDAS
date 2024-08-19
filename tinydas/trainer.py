from typing import List, Callable

from tinygrad import GlobalCounters, TinyJit
from tinygrad.nn import Tensor, state
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import colored

from tinydas.dataloader import DataLoader
from tinydas.early_stopping import EarlyStopping
from tinydas.models.base import BaseAE
from tinydas.plots import plot_loss
from tinydas.utils import save_model, printing, clip_grad_norm
from tinydas.timer import Timer


class Trainer:
    def __init__(
        self,
        model: BaseAE,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        **kwargs,
    ) -> None:
        self.shape = (kwargs["mod"]["M"], kwargs["mod"]["N"])
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optim = optimizer
        self.best_loss = float("inf")
        self.epochs = kwargs["epochs"] if kwargs["epochs"] else 10
        self.train_losses = [float(0)] * self.epochs
        self.val_losses = [float(0)] * self.epochs
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

    @TinyJit
    def val_step(self, x: Tensor, f: Callable[[Tensor], Tensor]) -> Tensor:
        x = f(x)
        loss = self.model.criterion(x)["loss"]
        return loss.realize()

    def train(self):
        print(colored(f"Starting training {self.model.__class__.__name__} with {self.epochs} epochs", 'yellow'))
        reshape_fn = lambda x: x.reshape(-1, 1, self.shape[0], self.shape[1]) if self.model.convolutional else x.reshape(-1, self.shape[0] * self.shape[1])
 
        for epoch in range(self.epochs):
            GlobalCounters.reset()

            # Training
            Tensor.training = True
            running_train_loss = 0.0
            with Timer() as train_t:
                for data in self.train_dataloader:
                    running_train_loss += self.train_step(data, reshape_fn).item()
            train_loss = running_train_loss / len(self.train_dataloader)
            self.train_losses.append(train_loss)

            # Validation
            Tensor.training = False
            running_val_loss = 0.0
            with Timer() as val_t:
                for data in self.val_dataloader:
                    running_val_loss += self.val_step(data, reshape_fn).item()
            val_loss = running_val_loss / len(self.val_dataloader)
            self.val_losses.append(val_loss)

            printing(epoch, self.epochs, train_loss, train_t.interval, val_loss, val_t.interval)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_model(self.model, "best")

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Best validation loss: {self.best_val_loss:.9f}")
        print(f"Train - Max loss: {max(self.train_losses):.9f}, Min loss: {min(self.train_losses):.9f}")
        print(f"Val - Max loss: {max(self.val_losses):.9f}, Min loss: {min(self.val_losses):.9f}")
        save_model(self.model, "final")
        plot_loss(self.train_losses, self.val_losses, self.model)
