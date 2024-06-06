from typing import List

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

