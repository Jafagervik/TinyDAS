from tinygrad import TinyJit
from tinygrad.nn import Tensor
from .dataloader import DataLoader
from .loss import mse
from tqdm import trange
from typing import List
from tqdm import trange


class Trainer: 
    def __init__(self, model, dataloader: DataLoader, optimizer, gpus: List[str]) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optim = optimizer
        self.gpus = gpus
        self.best_loss = float("inf")
        self.losses = []

    @TinyJit
    def _run_batch(self, data: Tensor, **kwargs):
        with Tensor.train():
            self.optim.zero_grad()

            samples = Tensor.randint(512, high=X_train.shape[0])
            Xt = data.shard_(self.gpus, axis=0)
            
            loss = self.model.loss_function().backward()

            self.optim.step()
            return loss

    def _run_epoch(self, epoch: int, **kwargs): 
        print(f"Epoch {epoch + 1}")
        for batch in self.dataloader:
            loss = self._run_batch(batch, epoch, **kwargs)
        self.losses.append(loss.item())

    def train(self, epochs: int, **kwargs):
        epochs = kwargs["epochs"]

        print("Training...")
        self.model.train()
        for epoch in (t:= trange(epochs)):
            self._run_epoch(epoch, **kwargs)
        self._save_checkpoint(epochs, final=True)
        pass

    def test(self):
        pass

    def load(self):
        pass
    
    def _save(self): pass
        pass
