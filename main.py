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
from tinydas.constants import SAMPLING_FREQ, DURATION

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
    print("Train mode")
    config = get_config(args.model)
    seed_all(config["data"]["seed"])
    print("Got config")

    devices = get_gpus(args.gpus) #devices = ["CLANG"]
    for x in devices: Device[x]

    dl = get_data(devices, **config)
    print("Dataloader complete")

    model = select_model(args.model, devices, **config)

    if args.load: load_model(model)

    if config["data"]["half_prec"]:
        for x in nn.state.get_state_dict(model).values():
            x = x.float().half()

    if len(devices) > 1:
        for x in nn.state.get_state_dict(model).values():
            x.realize().to_(devices)

    params = nn.state.get_parameters(model)
    optim = select_optimizer(Opti.ADAM, params, **config["opt"])
    print("Optimizer set")

    trainer = Trainer(model, dl, optim, **config)
    trainer.train()

    plot_loss(trainer.losses, trainer.model, save=True)


def show_imgs(args, devices: List[str], filename: str = ""):
    config = get_config(args.model)
    model = select_model(args.model, devices, **config)
    load_model(model)

    filename = filename or "./infer/20190415_032000.hdf5"
    data = load_das_file_no_time(filename)
    data = minmax(data).cast(dtypes.float16)

    filename = format_filename(filename)

    plot_das_as_heatmap(
        data, filename, show=True, path=f"figs/{args.model}/before/{filename}.png"
    )

    reconstructed = model.predict(data)

    plot_das_as_heatmap(
        reconstructed, filename, show=True, path=f"figs/{args.model}/after/{filename}.png"
    )

    return data, reconstructed

def find_recons(args, devices: List[str], filename: str = ""):
    config = get_config(args.model)
    model = select_model(args.model, devices, **config)
    load_model(model)

    data = load_das_file_no_time(filename)
    data = minmax(data).cast(dtypes.float16)

    filename = format_filename(filename)

    reconstructed = model.predict(data)

    return data, reconstructed

def anomaly_mode(args):
    # stream or img mode

    config = get_config(args.model)
    filename = "/cluster/home/jorgenaf/TinyDAS/data/20200302_081015.hdf5"

    predict_file(filename, args.model, **config)

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
