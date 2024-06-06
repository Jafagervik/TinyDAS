from tinygrad import Tensor, nn
from tinygrad.device import Device

from tinydas.dataloader import DataLoader
from tinydas.dataset import DataSet
from tinydas.trainer import Trainer
from tinydas.utils import *


class Model:
    def __init__(self):
        self.layers = [
            nn.Linear(2137 * 7500, 10),
            Tensor.gelu,
            nn.Linear(10, 2137 * 7500),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


def main():
    args = parse_args()
    config = get_config(args.filename)

    seed_all(config["seed"])

    GPUS = get_gpus(config["gpus"])
    print(f"Training on {GPUS}")

    # ds = DataSet()
    dl = DataLoader(
        dataset=DataSet("./data", n=config["n"]), batch_size=config["batch_size"]
    )

    model = Model()
    # we put a copy of the model on every GPU
    if len(GPUS) > 1 and any(GPUS) == "CLANG":
        for _, x in nn.state.get_state_dict(model).items():
            x.to_(GPUS)
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=config["lr"])

    trainer = Trainer(model, dataloader=dl, optimizer=opt, devices=GPUS, **config)
    trainer.train()


if __name__ == "__main__":
    main()
