from tinygrad import Tensor, nn
from tinygrad.device import Device

from tinydas.dataloader import DataLoader
from tinydas.dataset import Dataset
from tinydas.models.ae import AE
from tinydas.trainer import Trainer
from tinydas.utils import *


def main():
    model = AE()

    dataset = Dataset(n=10)
    dl = DataLoader(dataset, batch_size=2)
    optim = nn.optim.AdamW(nn.state.get_parameters(model), lr=0.5)
    devices = ["CLANG"]

    trainer = Trainer(
        model, dl, optim, devices, epochs=10, patience=5, min_delta=0.0, load=False
    )

    trainer.train()


if __name__ == "__main__":
    main()
