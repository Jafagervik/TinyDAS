import os
from typing import List, Optional

#import matplotlib
import matplotlib.pyplot as plt
from tinygrad import Tensor
import numpy as np
from datetime import timedelta, datetime

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
    if show: plt.show()
    if save:
        plt.savefig(os.path.join("figs", model.__class__.__name__.lower(), "loss.png"))


def plot_das_as_heatmap(
    das: np.ndarray, filename: str, show: bool = True, path: Optional[str] = None, num_ticks: int = 6
) -> None:
    ts = filename.split("/")[-1].split(".")[0]
    ts = "".join(ts.split("_"))
    print(f"{ts=}")
    start_timestamp =  datetime.strptime(ts, '%Y%m%d%H%M%S')
    
    # Calculate the duration and generate time labels
    total_duration = 5  # seconds
    total_samples = das.shape[0]
    time_per_sample = total_duration / total_samples
    
    # Generate time labels for the y-axis
    time_labels = [start_timestamp + timedelta(seconds=i*time_per_sample) for i in range(total_samples)]
    
    tick_indices = np.linspace(0, total_samples - 1, num_ticks).astype(int)
    tick_labels = [time_labels[i].strftime('%H:%M:%S.%f')[:-3] for i in tick_indices]
    
    # Plot the heatmap
    plt.figure(figsize=(10, 5))
    plt.imshow(das, aspect="auto", cmap="seismic")
    plt.colorbar()
    
    # Set the x-axis and y-axis labels
    plt.xlabel("Channel")
    plt.ylabel("Time")
    plt.title("Das Matrix as Heatmap")
    
    # Set y-ticks with time labels
    plt.yticks(tick_indices, tick_labels)
    
    if show: plt.show()
    if path is not None: plt.savefig(path)


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
