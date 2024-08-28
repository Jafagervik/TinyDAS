import os
from typing import List, Optional

import matplotlib.pyplot as plt
from tinygrad import Tensor
import numpy as np
from datetime import timedelta, datetime

from tinydas.models.base import BaseAE

def plot_loss(
    train_losses: List[float], 
    val_losses: List[float], 
    model: BaseAE, 
    show: bool = False, 
    save: bool = True,
) -> None:
    plt.figure(figsize=(10, 5))
    epochs = np.arange(start=1, stop=len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model.__class__.__name__} - Training and Validation Loss")
    plt.legend()

    if save:
        save_dir = os.path.join("figs", model.__class__.__name__.lower())
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "loss.png"))
    
    if show:
        plt.show()
    
    plt.close()


def plot_das_as_heatmap(
    das: np.ndarray, 
    filename: str, 
    show: bool = True, 
    path: Optional[str] = None, 
    num_ticks: int = 6, 
    vmin:Optional[int] = None, 
    vmax: Optional[int] = None
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
    #time_labels = [start_timestamp + timedelta(seconds=i*time_per_sample) for i in range(total_samples)]
    
    #tick_indices = np.linspace(0, total_samples - 1, num_ticks).astype(int)
    #tick_labels = [time_labels[i].strftime('%H:%M:%S.%f')[:-3] for i in tick_indices]
    
    # Plot the heatmap
    plt.figure(figsize=(10, 5))
    if vmin is not None and vmax is not None: 
        plt.imshow(das, aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)
    elif vmin is not None:
        plt.imshow(das, aspect="auto", cmap="seismic", vmin=vmin)
    elif vmax is not None:
        plt.imshow(das, aspect="auto", cmap="seismic", vmax=vmax)
    else:
        plt.imshow(das, aspect="auto", cmap="seismic")
    plt.axis('off')
    #plt.colorbar()
    
    # Set the x-axis and y-axis labels
    #plt.xlabel("Channel")
    #plt.ylabel("Time")
    #plt.title(f"{ts}")

    # Set y-ticks with time labels
    #plt.yticks(tick_indices, tick_labels)
    
    if path is not None: 
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
    if show: plt.show()
    plt.close()

def visualize_latent_space(model, data, num_points=1000):
    # Ensure the model is in evaluation mode
    Tensor.no_grad = True
    
    # Get encodings
    encodings = model.encoder(data[:num_points]).detach().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(encodings[:, 0], encodings[:, 1], alpha=0.5)
    plt.title('Latent Space Visualization')
    plt.xlabel('First Latent Dimension')
    plt.ylabel('Second Latent Dimension')
    plt.savefig(f'figs/{model.name}/latent_space.png')
    plt.close()
    Tensor.no_grad = False
