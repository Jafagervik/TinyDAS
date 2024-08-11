import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tinydas.early_stopping import EarlyStopping
from tinydas.timer import Timer
from ptdas.utils import plot_losses, paramsize, weights_init
import torch.nn.functional as F

def train(model, dataset, num_epochs, batch_size, learning_rate, device, kl_scale):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model_name = model.__class__.__name__.lower()
    
    num_gpus = torch.cuda.device_count()
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))

    if num_gpus > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    #model.apply(weights_init)

    ps = paramsize(model)
    print(f'Model size: {ps:.3f}MB')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    #scaler = torch.amp.GradScaler("cuda")
    es = EarlyStopping(patience=3, min_delta=0.001)
    
    tot_losses, mse_losses, kld_losses = [], [], []
    best_loss = float('inf')
    
    print("Starting training")
    for epoch in range(num_epochs):
        total_loss, total_bce, total_kld = 0, 0, 0
        
        with Timer() as t:
            for batch in dataloader:
                images = batch.to(device)
                optimizer.zero_grad()

                recon_images, mu, logvar = model(images)
                #kl_scale = min(epoch / 100, 1.0) 
                loss, bce, kld = loss_function(recon_images, images, mu, logvar, kl_scale)
                
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_bce += bce.item()
                total_kld += kld.item()

        
            avg_loss = total_loss / len(dataloader)
            avg_bce = total_bce   / len(dataloader)
            avg_kld = total_kld   / len(dataloader)

        scheduler.step(avg_loss)
        
        tot_losses.append(avg_loss)
        mse_losses.append(avg_bce)
        kld_losses.append(avg_kld)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]: '
              f'Time: {t.interval:.2f}s, '
              f'Avg Loss: {avg_loss:.4f}, '
              f'Avg MSE: {avg_bce:.4f}, '
              f'Kld Weight: {kl_scale:.4f}, '
              f'Avg KLD: {avg_kld:.4f} weight')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.module.state_dict(), f'checkpoints/{model_name}/best.pth')
        
        es(avg_loss)
        if es.early_stop:
            print("Stopping training...")
            break
    
    plot_losses(tot_losses, mse_losses, kld_losses, model_name)
    print(f"Final loss: {tot_losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f}")


def loss_function(recon_x, x, mu, logvar, kl_weight, eps=1e-8):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean') 
    
    # KL divergence loss
    kl_loss = torch.sum(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, dim=1), dim=0) 
    
    # Total loss (ELBO)
    total_loss = reconstruction_loss +  (kl_loss * kl_weight)
    
    return total_loss, reconstruction_loss, kl_loss * kl_weight