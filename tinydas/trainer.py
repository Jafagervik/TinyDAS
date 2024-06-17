from typing import List

from tinygrad import GlobalCounters, TinyJit, dtypes
from tinygrad.nn import Tensor
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import colored
from tqdm import trange

from tinydas.dataloader import DataLoader
from tinydas.early_stopping import EarlyStopping
from tinydas.models.base import BaseAE
from tinydas.plots import plot_loss
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
        self.shape = (kwargs["mod"]["M"], kwargs["mod"]["N"])
        self.model = model
        self.dataloader = dataloader
        self.optim = optimizer
        self.devices = devices
        self.best_loss = float("inf")
        self.epochs = kwargs["epochs"] if kwargs["epochs"] else 10
        self.losses = [float(0)] * self.epochs
        self.early_stopping = EarlyStopping(
            kwargs["es"]["patience"], kwargs["es"]["min_delta"]
        )

    def _run_epoch(self) -> Tensor:
        Tensor.training = True
        self.optim.zero_grad()
        samples = Tensor.randint(
            self.dataloader.batch_size, high=self.dataloader.num_samples
        )
        print(samples.numpy())
        x = self.dataloader.data[samples].shard_(self.devices, axis=0)

        if self.model.convolutional:
            # [BS, C, M, N]
            x = x.reshape(-1, 1, self.shape[0], self.shape[1])  
        else:
            # [BS, M, N]
            x = x.reshape(-1, self.shape[0] * self.shape[1])

        loss_dict = self.model.criterion(x)
        loss = loss_dict["loss"]

        loss.backward()
        self.optim.step()

        return loss
        # running_loss += loss.item()

        # running_loss /= self.dataloader.num_samples / self.dataloader.batch_size
        # return Tensor(running_loss)

    @TinyJit
    def train_step(self, x: Tensor) -> Tensor:
        Tensor.training = True
        self.optim.zero_grad()
        print("inside")

        if self.model.convolutional:
            # [BS, C, M, N]
            x = x.reshape(-1, 1, self.shape[0], self.shape[1])  
        else:
            # [BS, M, N]
            x = x.reshape(-1, self.shape[0] * self.shape[1])

        loss_dict = self.model.criterion(x)
        loss = loss_dict["loss"]

        loss.backward()
        self.optim.step()

        return loss.realize()

    def train(self):
        print(colored(f"Starting training {self.model.__class__.__name__} with {self.epochs} epochs", 'red'))

        for epoch in range(self.epochs):
        #for epoch in (t := trange(self.epochs)):
            Tensor.training = True
            print(colored(f"Epoch {epoch + 1}/{self.epochs}", "green"))
            #loss = self._run_epoch(epoch)

            for data in self.dataloader:
                GlobalCounters.reset()
                loss = self.train_step(data)
            self.losses[epoch] = loss.item()

            #t.set_description(f"Epoch: {epoch + 1} | Loss: {loss.item():.4f}")
            print(f"Loss: {loss.item():.4f}")

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                save_model(self.model)

            self.early_stopping(loss.item())
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                save_model(self.model, final=True)
                plot_loss(self.losses, self.model)
                return

        print(f"Max loss: {max(self.losses):.4f}, Min loss: {min(self.losses):.4f}")
        save_model(self.model, final=True)
