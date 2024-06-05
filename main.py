from typing import Any
from src.dataset import DataSet
from src.dataloader import DataLoader
from src.utils import *
from src.models.ae import AE
from tinygrad import TinyJit, GlobalCounters
from tqdm import trange

class Model: 
    def __init__(self):
        self.layers = [
            nn.Linear(2137 * 7500, 1),
            Tensor.sigmoid
        ]

    def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)


def main():
    DEBUG = 0

    args = parse_args()
    config = get_config(args.filename)

    seed_all(config["seed"])

    GPUS = get_gpus(config["gpus"])

    ds = DataSet()
    dl = DataLoader(dataset=ds, batch_size=config["batch_size"])

    print(f"Training on {GPUS}")

    model = Model()
    for k, x in nn.state.get_state_dict(model).items(): x.to_(GPUS)  # we put a copy of the model on every GPU
    opt = nn.optim.Adam(nn.state.get_parameters(model))

    @TinyJit
    def step() -> Tensor:
        with Tensor.train():
            opt.zero_grad()

            for data, _ in dl:
                x = data.shard_(GPUS, axis=0).reshape(-1, 2137 * 7500)

                loss = model(x).sub(x).square().mean().backward()
                opt.step()
            return loss
    
    for epoch in (t:=trange(config["epochs"])):
        GlobalCounters.reset()
        loss = step()
        t.set_description(f"loss: {loss.item():6.2f}")



if __name__ == "__main__": main()