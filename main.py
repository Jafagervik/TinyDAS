#!/usr/bin/env python
from tinygrad import nn

from tinydas.anomalies import predict_file
from tinydas.dataloader import DataLoader
from tinydas.dataset import Dataset
from tinydas.plots import plot_loss
from tinydas.selections import select_model, select_optimizer
from tinydas.trainer import Trainer
from tinydas.utils import *

def train_mode(args):
    """Train the model on the dataset."""

    debug = False
    devices = get_gpus(2)
    config = get_config(args.model)
    seed_all(config["seed"])

    # TODO: Use multiple GPUs
    #devices = ["CUDA:0"]

    if debug: 
        print(devices)

    dataset = Dataset(n=config["nfiles"])
    dl = DataLoader(dataset, batch_size=config["batch_size"])

    if debug: 
        print(dataset.shape)

    if debug: 
        print(dl.num_samples)

    model = select_model(args.model, **config)
    
    if args.load == True:
        config["load"] = True
        load_model(model)

    if debug: 
        print(model)

    if len(devices) > 1:
        for _, x in nn.state.get_state_dict(model).items():
            x.to_(devices)

    params = nn.state.get_parameters(model)
    optim = nn.optim.Adam(params, lr=config["lr"])

    # optim = select_optimizer(config["optimizer"], params, config["lr"])

    trainer = Trainer(model, dl, optim, devices, **config)

    if debug:
        print(trainer.best_loss)

    trainer.train()

    #plot_loss(trainer, save=True)


def anomaly_mode(args):
    # stream or img mode

    config = get_config(args.model)
    filename = "./data/20200301_000015.hdf5"

    predict_file(filename, **config)

    # img mode

    # das_img = None

    # model = load_model()

    # anomalies = find_anomalies(das_img, model)

    # plot anomalies

    # find anomaly scores and so on


def main():
    args = parse_args()

    match args.type:
        case "train":
            train_mode(args)
        case "detect":
            anomaly_mode(args)
        case _:
            print("Invalid mode, please select train or detect.")
            exit(1)


if __name__ == "__main__":
    main()
