from tinygrad import nn

from tinydas.dataloader import DataLoader
from tinydas.dataset import Dataset
from tinydas.models.ae import AE
from tinydas.models.cnn_ae import CNNAE
from tinydas.models.vae import VAE
from tinydas.trainer import Trainer
from tinydas.utils import *


def train_mode(args):
    """Train the model on the dataset."""

    config = get_config(args.model)
    seed_all(config["seed"])

    devices = ["CLANG"]

    match args.model:
        case "ae":
            model = AE(**config)
        case "vae":
            model = VAE(**config)
        case "cnnae":
            model = CNNAE(**config)
        case _:
            model = AE(**config)

    dataset = Dataset(n=config["nfiles"])
    dl = DataLoader(dataset, batch_size=config["batch_size"])

    optim = nn.optim.AdamW(nn.state.get_parameters(model), lr=config["lr"])

    trainer = Trainer(model, dl, optim, devices, **config)

    trainer.train()


def anomaly_mode(args):
    _ = args
    print("TBI")


def main():
    args = parse_args()

    match args.type:
        case "train":
            train_mode(args)
        case "detect":
            anomaly_mode(args)
        case _:
            print("Invalid mode")
            exit(1)


if __name__ == "__main__":
    main()
