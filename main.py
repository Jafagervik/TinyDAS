from src.trainer import Trainer
from src.data import DataLoader
from src.plots import *
from src.utils import *
from src.models.base import AE


def train():
    DEBUG = 0

    args = parse_args()
    config = get_config(args.filename)

    seed_all(config["seed"])

    GPUS = get_gpus(config["gpus"])

    print(f"Training on {GPUS}")

    model = AE(
        M=config["model_params"]["M"],
        N=config["model_params"]["N"],
        latent_dim=config["model_params"]["latent_dim"],
        hidden_dim=config["model_params"]["hidden_dim"],
    )

    for _, x in nn.state.get_state_dict(model).items(): x.to_(GPUS)  # we put a copy of the model on every GPU
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=config["lr"])

    data = None
    dl = DataLoader(data)

    if DEBUG > 1:
        print(data.shape)

    trainer = Trainer(model, dl, opt, GPUS)

    trainer.train(config["epochs"], **config)


def anom_detect():
    pass

if __name__ == "__main__":
    train()