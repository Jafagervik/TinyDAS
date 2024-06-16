#!/usr/bin/env python
from tinygrad import nn

from tinydas.anomalies import predict_file
from tinydas.dataloader import DataLoader
from tinydas.dataset import Dataset
from tinydas.plots import plot_das_as_heatmap, plot_loss
from tinydas.selections import select_model, select_optimizer
from tinydas.trainer import Trainer
from tinydas.utils import *


def train_mode(args):
    """Train the model on the dataset."""

    debug = False
    config = get_config(args.model)
    seed_all(config["data"]["seed"])

    # devices = get_gpus(config["gpus"])
    devices = ["CLANG"]

    if debug:
        print(devices)

    dataset = Dataset(n=config["data"]["nfiles"])
    if debug:
        print(dataset)
        get_size_in_gb(dataset.data["data"])
    d = dataset.data["data"]
    print(d.min().item(), d.max().item())

    dl = DataLoader(dataset, batch_size=config["data"]["batch_size"], devices=devices)

    if debug:
        print(dl.num_samples)

    model = select_model(args.model, **config)

    if args.load == True:
        config["load"] = True
        load_model(model)

    if debug:
        print(model)

    if config["data"]["half_prec"]:
        for x in nn.state.get_state_dict(model).values():
            x = x.float().half()

    if len(devices) > 1:
        for x in nn.state.get_state_dict(model).values():
            x.to_(devices)

    params = nn.state.get_parameters(model)
    optim = nn.optim.Adam(
        params,
        lr=config["opt"]["lr"],
        b1=config["opt"]["b1"],
        b2=config["opt"]["b2"],
    )

    # optim = select_optimizer(config["optimizer"], params, config["lr"])

    trainer = Trainer(model, dl, optim, **config)

    if debug:
        print(trainer.best_loss)

    trainer.train()

    plot_loss(trainer.losses, trainer.model, save=True)


def show_imgs():
    import h5py

    with h5py.File("./data/20200301_001650.hdf5", "r") as f:
        data = np.array(f["raw"][:], dtype=np.float32).T
        print(data[0, 0])
        print(data[100, 100])
        print(data[200, 200])
        print(data[600, 2100])

        t = Tensor(data)
        t = minmax(t)

        plot_das_as_heatmap(t.numpy())


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
