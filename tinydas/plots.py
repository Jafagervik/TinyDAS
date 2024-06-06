from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(losses: List[float], show: bool = True) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    if show:
        plt.show()
    plt.savefig("loss.png")


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
    from tinygrad import Tensor
    from tinygrad.dtype import dtypes

    t = Tensor.randn(28, 28, dtype=dtypes.float32)

    plot_das_as_heatmap(t.numpy())
