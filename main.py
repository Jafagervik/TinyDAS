#!/usr/bin/env python
from tinygrad import nn, Device, dtypes

from tinydas.anomalies import predict_file
from tinydas.dataloader import DataLoader
from tinydas.dataset import Dataset
from tinydas.plots import plot_das_as_heatmap, plot_loss
from tinydas.selections import Opti, select_model, select_optimizer
from tinydas.trainer import Trainer
from tinydas.utils import *
from tinydas.enums import Normalization

def get_data(devices: List[str], **config) -> DataLoader:
    dataset = Dataset(
        n=config["data"]["nfiles"],
        normalize=Normalization.MINMAX,
        dtype=dtypes.float16 if config["data"]["half_prec"] else dtypes.float32  
    )
    dl = DataLoader(
        dataset, 
        batch_size=config["data"]["batch_size"], 
        devices=devices, 
        num_workers=config["data"]["num_workers"]
    )
    return dl

def train_mode(args):
    config = get_config(args.model)
    seed_all(config["data"]["seed"])

    devices = get_gpus(args.gpus) #devices = ["CLANG"]
    for x in devices: Device[x]

    dl = get_data(devices, **config)

    model = select_model(args.model, **config)

    if args.load: load_model(model)

    if config["data"]["half_prec"]:
        print("F16")
        for x in nn.state.get_state_dict(model).values():
            x = x.float().half()

    if len(devices) > 1:
        for x in nn.state.get_state_dict(model).values():
            x.realize().to_(devices)

    params = nn.state.get_parameters(model)
    optim = select_optimizer(Opti.ADAM, params, **config["opt"])

    trainer = Trainer(model, dl, optim, devices, **config)
    trainer.train()

    plot_loss(trainer.losses, trainer.model, save=True)


def show_imgs(args):
    import h5py

    config = get_config(args.model)
    model = select_model(args.model, **config)

    filename = "./data/20200301_001650.hdf5"

    with h5py.File(filename, "r") as f:
        data = Tensor(f["raw"][:], dtype=dtypes.float16).T
        plot_das_as_heatmap(
            data.numpy(), show=False, path=f"figs/{args.model}/before.png"
        )

        reconstructed = model.predict(data)

        plot_das_as_heatmap(
            reconstructed.numpy(), show=False, path=f"figs/{args.model}/after.png"
        )


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
        case "show":
            show_imgs(args)
        case _:
            print("Invalid mode, please select train or detect.")
            exit(1)


if __name__ == "__main__":
    main()
