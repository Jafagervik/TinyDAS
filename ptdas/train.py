import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tinydas.early_stopping import EarlyStopping
from tinydas.timer import Timer
from ptdas.utils import plot_losses, paramsize, weights_init
import torch.nn.functional as F

def train(model, dataset, num_epochs, batch_size, learning_rate, device, kl_scale):
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model_name = model.__class__.__name__.lower()
    
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(torch.cuda.get_device_name(i))

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    ps = paramsize(model)
    print(f'Model size: {ps:.3f}MB')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    es = EarlyStopping(patience=5, min_delta=0.001)
    
    train_losses, val_losses = [], []
    train_mse_losses, val_mse_losses = [], []
    train_kld_losses, val_kld_losses = [], []
    best_val_loss = float('inf')
    
    print("Starting training")
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss, total_bce, total_kld = 0, 0, 0
        
        with Timer() as t:
            for batch in train_dataloader:
                images = batch.to(device)
                optimizer.zero_grad()

                recon_images, mu, logvar = model(images)
                loss, bce, kld = loss_function(recon_images, images, mu, logvar, kl_scale)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_bce += bce.item()
                total_kld += kld.item()

            avg_train_loss = total_loss / len(train_dataloader)
            avg_train_bce = total_bce / len(train_dataloader)
            avg_train_kld = total_kld / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss, total_val_bce, total_val_kld = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch.to(device)
                recon_images, mu, logvar = model(images)
                val_loss, val_bce, val_kld = loss_function(recon_images, images, mu, logvar, kl_scale)
                
                total_val_loss += val_loss.item()
                total_val_bce += val_bce.item()
                total_val_kld += val_kld.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_bce = total_val_bce / len(val_dataloader)
        avg_val_kld = total_val_kld / len(val_dataloader)

        scheduler.step(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_mse_losses.append(avg_train_bce)
        val_mse_losses.append(avg_val_bce)
        train_kld_losses.append(avg_train_kld)
        val_kld_losses.append(avg_val_kld)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]: '
              f'Time: {t.interval:.2f}s, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Train MSE: {avg_train_bce:.4f}, '
              f'Val MSE: {avg_val_bce:.4f}, '
              f'KL Weight: {kl_scale:.4f}, '
              f'Train KLD: {avg_train_kld:.4f}, '
              f'Val KLD: {avg_val_kld:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
                       f'checkpoints/{model_name}/best.pth')
        
        es(avg_val_loss)
        if es.early_stop:
            print("Early stopping triggered")
            break
    
    plot_losses(train_losses, val_losses, model_name)
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")


def loss_function(recon_x, x, mu, logvar, kl_weight, eps=1e-8):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean') 
    
    # KL divergence loss
    #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1)
    kl_loss = torch.sum(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, dim=1), dim=0) / x.size(0)
    
    # Total loss (ELBO)
    total_loss = reconstruction_loss +  (kl_loss * kl_weight) / x.size(0)
    
    return total_loss, reconstruction_loss, kl_loss * kl_weight / x.size(0)
