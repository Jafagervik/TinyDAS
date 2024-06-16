from typing import List

from tinygrad import GlobalCounters, TinyJit, dtypes
from tinygrad.nn import Tensor
from tinygrad.nn.optim import Optimizer
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
        # devices: List[str],
        **kwargs,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optim = optimizer
        # self.devices = devices
        self.best_loss = float("inf")
        self.epochs = kwargs["epochs"] if kwargs["epochs"] else 10
        self.losses = [float(0)] * self.epochs
        self.early_stopping = EarlyStopping(
            kwargs["es"]["patience"], kwargs["es"]["min_delta"]
        )

    @TinyJit
    def _run_epoch(self):  # -> Tensor:
        Tensor.training = True
        samples = Tensor.randint(
            self.dataloader.batch_size, high=self.dataloader.num_samples
        )
        x = self.dataloader.data[samples]
        # running_loss = 0.0
        # for x in self.dataloader:
        x = x.reshape(-1, 625 * 2137)

        self.optim.zero_grad()

        loss_dict = self.model.criterion(x)
        loss = loss_dict["loss"]

        loss.backward()
        self.optim.step()

        return loss
        # running_loss += loss.item()

        # running_loss /= self.dataloader.num_samples / self.dataloader.batch_size
        # return Tensor(running_loss)

    @TinyJit
    def step(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, 625 * 2137)

        loss_dict = self.model.criterion(x)
        loss = loss_dict["loss"]

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

    def train(self):
        print("Starting training...")
        with Tensor.train():
            for epoch in (t := trange(self.epochs)):
                GlobalCounters.reset()
                # loss = self._run_epoch()
                rl = 0.0
                for data in self.dataloader:
                    loss = self.step(data)
                    rl += loss.item()
                self.losses[epoch] = rl / 10.0

                t.set_description(f"Epoch: {epoch + 1} | Loss: {rl:.4f}")

                if rl < self.best_loss:
                    self.best_loss = rl
                    save_model(self.model)

                self.early_stopping(rl)
                if self.early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    save_model(self.model, final=True)
                    plot_loss(self.losses, self.model)

                    break
            save_model(self.model, final=True)
