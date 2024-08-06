import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List
import os
import numpy as np
import h5py
from tinydas.early_stopping import EarlyStopping
from tinydas.timer import Timer
#from tinydas.plots import plot_das_as_heatmap
#from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# SAVING AND LOADING
# https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html

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

# Hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 500
N = 25600
beta = 0.1
kl_scale = 1e-3

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def paramsize(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2
    
    
# Custom Dataset (based on your provided class)
class DASDataset(Dataset):
    def __init__(self, img_dir: str = "./data", n: int = 512):
        self.img_dir = img_dir
        self.n = n
        self.filenames = self._get_filenames(n)

    def __len__(self) -> int: return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        data = self.load_das_file_no_time(filename)
        data = self._apply_normalization(data)
        
        return torch.from_numpy(data).to(torch.float16).cpu()

    def _get_filenames(self, n: Optional[int]) -> List[str]:
        filenames = [entry.path for entry in os.scandir(self.img_dir)]
        if n is not None:
            filenames = filenames[:n]
        return filenames

    def load_das_file_no_time(self, filename: str) -> np.ndarray:
        with h5py.File(filename, "r") as f:
            data = np.array(f["raw"][:]).T
        return data
    
    def _apply_normalization(self, data: np.ndarray) -> np.ndarray:
        return (data - np.mean(data)) / np.std(data)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2137 * 625, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(128, 32)
        self.fc_logvar = nn.Linear(128, 32)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2137 * 625),
            nn.Sigmoid(),
            nn.Unflatten(1, (625, 2137))  # Reshape to original dimensions
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x.float(), x.float(), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().clamp(max=10))
    return BCE, KLD

def plot_losses(total_loss, mse_loss, kld_loss, model, show=False, save=True):
    plt.figure(figsize=(12, 6))
    epochs = np.arange(start=1, stop=len(total_loss) + 1)

    plt.plot(epochs, total_loss, label='Total Loss', color='blue')
    plt.plot(epochs, mse_loss, label='MSE Loss', color='red')
    plt.plot(epochs, kld_loss, label='KLD Loss', color='green')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model.__class__.__name__} - Losses vs Epochs")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if show: plt.show()
    if save:
        save_dir = os.path.join("figs", model.__class__.__name__.lower())
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "losses.png"))

def main():
    # Create dataset and dataloader
    dataset = DASDataset(img_dir='/cluster/home/jorgenaf/TinyDAS/data', n=N)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Dataloader done")

    num_gpus = torch.cuda.device_count()
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))

    model = VAE()
    #model = torch.compile(model)

    print("Model compiled")
    if num_gpus > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    #model.apply(weights_init)

    #summary(model, input_size=(batch_size, 625, 2137))


    ps = paramsize(model)
    print(f'Model size: {ps:.3f}MB')
    
    es = EarlyStopping(patience=5, min_delta=0.0002)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda")

    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    tot_losses = []
    mse_losses = []
    kld_losses = []

    print("Starting training")
    best_loss = float('inf')
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_bce = 0
        total_kld = 0
        
        with Timer() as t:
            for batch in dataloader:
                images = batch.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    recon_images, mu, logvar = model(images)
                    
                    bce, kld = loss_function(recon_images, images, mu, logvar)
                    loss = bce + kld * kl_scale
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                #scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                total_bce += bce.item()
                total_kld += kld.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_bce = total_bce / len(dataloader)
        avg_kld = total_kld / len(dataloader)
        
        tot_losses.append(avg_loss)
        mse_losses.append(avg_bce)
        kld_losses.append(avg_kld)

        #scheduler.step(avg_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]: '
            f'Time: {t.interval:.2f}s, '
            f'Loss: {avg_loss:.4f}, '
            f'MSE: {avg_bce:.4f}, '
            f'KLD: {avg_kld:.4f} without weight')
        
        if (avg_loss < best_loss): 
            best_loss = avg_loss
            torch.save(model.module.state_dict(), f'vae_{best_loss:.5f}.pth')

        es(avg_loss)
        if es.early_stop:
            plot_losses(tot_losses, mse_losses, kld_losses, model)
            print("Stopping training...")
            return
            
    
    plot_losses(tot_losses, mse_losses, kld_losses, model)
    print(f"Final loss: {tot_losses[-1]:.4f}")
    print("Complete")

    

def load_das_file_no_time(filename: str) -> np.ndarray:
    with h5py.File(filename, "r") as f:
        data = np.array(f["raw"][:]).T
    return data

def test(): 
    # Directory containing the infer files
    infer_dir = '/cluster/home/jorgenaf/TinyDAS/infer'

    # Load the model
    model = VAE().to(device)
    name = "meme.pth"
    sd = torch.load(name)
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for filename in os.listdir(infer_dir)[:2]:
                file_path = os.path.join(infer_dir, filename)
                
                data = load_das_file_no_time(file_path)
                
                data = (data - np.mean(data)) / np.std(data)
                
                d = torch.from_numpy(data).to(torch.float16).unsqueeze(0).to(device)  # Add batch dimension

                out = model(d)[0]

                # Plot output
                plot_das_as_heatmap(
                    out.cpu().numpy().squeeze(),  # Remove batch dimension
                    filename,
                    show=False,
                    path=f"/cluster/home/jorgenaf/TinyDAS/figs/vae/after/{filename[:-5]}.png"
                )

                mse = F.mse_loss(d, out).item()  # Assuming the first element of out is the reconstruction
                print(f"File: {filename}, MSE: {mse}")

    print("Processing complete.")



if __name__ == '__main__':
    #test()
    plot_latent()
    #main()
