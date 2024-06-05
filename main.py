from typing import Any
from src.dataset import DataSet
from src.dataloader import DataLoader
from src.utils import *
from src.trainer import Trainer
from tinygrad import  Tensor
from tinygrad import nn

class Model: 
    def __init__(self):
        self.layers = [
            nn.Linear(2137 * 7500, 10),
            Tensor.gelu,
            nn.Linear(10, 2137 * 7500),
            Tensor.sigmoid
        ]

    def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)


def main():
    args = parse_args()
    config = get_config(args.filename)

    seed_all(config["seed"])

    GPUS = get_gpus(config["gpus"])

    #ds = DataSet()
    dl = DataLoader(dataset=DataSet(), batch_size=config["batch_size"])

    print(f"Training on {GPUS}")

    model = Model() 
    # we put a copy of the model on every GPU
    for _, x in nn.state.get_state_dict(model).items(): x.to_(GPUS) 
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=config["lr"])

    trainer = Trainer(model, dataloader=dl, optimizer=opt, devices=GPUS, **config)
    trainer.train()


if __name__ == "__main__": main()