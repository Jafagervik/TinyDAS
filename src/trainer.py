from tinygrad import TinyJit, GlobalCounters
from tinygrad.nn import Tensor
from .dataloader import DataLoader
from .losses import mse
from typing import List
from tqdm import trange
from tinygrad.nn.optim import Optimizer


class Trainer: 
    def __init__(self, model, dataloader: DataLoader, optimizer: Optimizer, devices: List[str], **kwargs) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optim = optimizer
        self.gpus = devices
        self.best_loss = float("inf")
        self.epochs = kwargs["epochs"] if kwargs["epochs"] else 10
        self.losses = [float(0)] * self.epochs

    @TinyJit
    def _run_epoch(self): 
        with Tensor.train():
            # OPTIM ZERO GRAD

            for data, _ in self.dataloader:
                self.optim.zero_grad()

                x = data.shard_(self.gpus, axis=0).reshape(-1, 2137 * 7500)
                # Mse
                loss = self.model(x).sub(x).square().mean().backward()
                self.optim.step()
            return loss

    def train(self):
        print("Starting training...")
        for epoch in (t := trange(self.epochs)):
            GlobalCounters.reset()
            loss = self._run_epoch()
            self.losses[epoch] = loss.item()
            t.set_description(f"Epoch: {epoch + 1} |> loss: {self.losses[epoch]:.2f}")