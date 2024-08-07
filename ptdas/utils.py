import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import os
from typing import Optional, List

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight) 
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def paramsize(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def plot_das_as_heatmap(
    das: np.ndarray, filename: str, show: bool = True, path: Optional[str] = None, num_ticks: int = 6
) -> None:
    ts = filename.split("/")[-1].split(".")[0]
    name = ts[:8]
    ts = "".join(ts.split("_"))
    print(f"{name=}")
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
    plt.title(f"DAS Data: {name}")
    
    # Set y-ticks with time labels
    plt.yticks(tick_indices, tick_labels)
    
    if show: plt.show()
    if path is not None: plt.savefig(path)

def plot_losses(total_loss, mse_loss, kld_loss, model_name, show=False, save=True):
    plt.figure(figsize=(12, 6))
    epochs = np.arange(start=1, stop=len(total_loss) + 1)

    plt.plot(epochs, total_loss, label='Total Loss', color='blue')
    #plt.plot(epochs, mse_loss, label='MSE Loss', color='red')
    #plt.plot(epochs, kld_loss, label='KLD Loss', color='green')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if show: plt.show()
    if save:
        save_dir = os.path.join("figs", model_name)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "losses.png"))