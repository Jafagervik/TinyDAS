import os
from typing import List, Optional

#import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#matplotlib.use("QT5Agg")

from tinydas.models.base import BaseAE


def plot_loss(
    losses: List[float], model: BaseAE, show: bool = False, save: bool = True  
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(start=1, stop=len(losses) + 1), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join("figs", model.__class__.__name__.lower(), "loss.png"))


def plot_das_as_heatmap(
    das: np.ndarray, show: bool = True, path: Optional[str] = None
) -> None:
    plt.figure(figsize=(10, 5))
    plt.imshow(das, aspect="auto", cmap="seismic")
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Time")
    plt.title("Das Matrix as Heatmap")
    if show:
        plt.show()
    if path is not None:
        plt.savefig(path)


if __name__ == "__main__":
    # from datetime import datetime

    import h5py

    from tinygrad import Tensor, dtypes

    with h5py.File("../data/20200301_001650.hdf5", "r") as f:
        data = Tensor(f["raw"][:], dtype=dtypes.float16).T
        plot_das_as_heatmap(data.numpy())

        # times = Tensor(np.array(f["timestamp"][:]))
        # print(data.shape)
        # timestamp = times.numpy()[0]
        # print(timestamp)
        # dt_object = datetime.fromtimestamp(timestamp)

        # print("Datetime object:", dt_object)
