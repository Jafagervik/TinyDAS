import os
import h5py
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tinydas.early_stopping import EarlyStopping

class MyTrainDataset(Dataset):
    def __init__(self, img_dir: str = "/cluster/home/jorgenaf/TinyDAS/data", n: int = 25600):
        self.img_dir = img_dir
        self.n = n
        self.filenames = self._get_filenames(n)

    def __len__(self) -> int: return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        data = self.load_das_file_no_time(filename)
        data = self._apply_normalization(data)
        
        data = torch.from_numpy(data).to(torch.float16).unsqueeze(0)
        return data

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
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Output: 312x1069
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 156x535
            nn.ReLU(),
        )
        
        # Calculate the size of the flattened feature map
        self.feature_size = 156 * 535 * 32
        
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.feature_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 312x1069
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # Output: 624x2137
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 32, 156, 535)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD, MSE, KLD

def main():
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set up the device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create model
    model = VAE().to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create DataLoader
    dataset = MyTrainDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)

    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()

    # Beta schedule
    beta_start = 0.0
    beta_end = 1.0
    n_epochs = 100
    beta_schedule = torch.linspace(beta_start, beta_end, n_epochs)

    best_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        total_mse = 0
        total_kld = 0

        beta = beta_schedule[epoch].item()

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            with autocast():
                recon_batch, mu, logvar = model(data)
                loss, mse, kld = loss_function(recon_batch, data, mu, logvar, beta)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()

        # Gather losses from all GPUs
        avg_loss = torch.tensor(total_loss / len(train_loader), device=device)
        avg_mse = torch.tensor(total_mse / len(train_loader), device=device)
        avg_kld = torch.tensor(total_kld / len(train_loader), device=device)

        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_kld, op=dist.ReduceOp.SUM)

        avg_loss /= world_size
        avg_mse /= world_size
        avg_kld /= world_size

        if rank == 0:
            print(f'Epoch {epoch}, Beta: {beta:.4f}, Avg Loss: {avg_loss.item():.4f}, '
                  f'Avg BCE: {avg_mse.item():.4f}, Avg KLD: {avg_kld.item():.4f}')

            # Save the best model based on training loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'beta': beta,
                }, 'best_model.pth')
                print(f'New best model saved with loss: {best_loss.item():.4f}')

    dist.destroy_process_group()

if __name__ == "__main__":
    main()