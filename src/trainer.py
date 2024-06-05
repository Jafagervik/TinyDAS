from tinygrad import TinyJit
from tinygrad.nn import Tensor
from .data import DataLoader
from .loss import mse
from tqdm import trange


class Trainer: 
    def __init__(self, model, dataloader: DataLoader, optimizer, gpus) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optim = optimizer
        self.gpus = gpus
        self.best_loss = float("inf")
        self.losses = []

    @TinyJit
    def _run_batch(self, **kwargs):
        with Tensor.train():
            self.optim.zero_grad()

            samples = Tensor.randint(512, high=X_train.shape[0])
            Xt, Yt = X_train[samples].shard_(self.gpus, axis=0), Y_train[samples].shard_(self.gpus, axis=0)  # we shard the data on axis 0
            # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
            loss = mse(self.model, Xt).backward()

            self.optim.step()
            return loss

    def _run_epoch(self, epoch: int, **kwargs): 
        print(f"Epoch {epoch + 1}")
        for batch in self.train_loader:
            for data in batch:
                self._run_batch(data, epoch, **kwargs)
            self.optimizer.step()

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

@TinyJit
def train_step() -> Tensor:
    pass


