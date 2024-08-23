from typing import List, Callable, Optional
import time

from tinydas.kl import kl_annealing
from tinydas.losses import mse
from tinygrad import GlobalCounters, TinyJit, dtypes
from tinygrad.nn import Tensor, state
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import colored

import numpy as np

from tinydas.dataloader import DataLoader
from tinydas.early_stopping import EarlyStopping
from tinydas.models.base import BaseAE
from tinydas.plots import plot_loss
from tinydas.utils import check_overflow, save_model, printing, clip_and_grad
from tinydas.timer import Timer
from tinydas.lr_schedule import LR_Scheduler, WarmupScheduler


class Trainer:
    def __init__(
        self,
        model: BaseAE,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LR_Scheduler] = None,
        **kwargs,
    ) -> None:
        self.shape = (kwargs["mod"]["M"], kwargs["mod"]["N"])
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optim = optimizer

        self.scheduler = scheduler #if self.model.__class__.__name__.lower() in ["vae", "cae"] else None

        self.best_loss = float("inf")
        self.epochs = kwargs["epochs"] if kwargs["epochs"] else 10
        self.train_losses = []
        self.val_losses = []
        self.early_stopping = EarlyStopping(
            kwargs["es"]["patience"], kwargs["es"]["min_delta"]
        )

        self.loss_scaler = kwargs["opt"]["loss_scale"]
        self.batch_size = kwargs["data"]["batch_size"]

    @TinyJit
    @Tensor.train()
    def train_step(self, input_batch: Tensor):
        self.optim.zero_grad()
        loss = self.model.criterion(input_batch)
        (loss * self.loss_scaler).backward()
    
        clip_and_grad(self.optim, self.loss_scaler)
       # global_norm = Tensor([0.0], dtype=dtypes.float32, device=self.optim.params[0].device).realize()
       # for param in self.optim.params: 
       #     if param.grad is not None:
       #         param.grad = param.grad / self.loss_scaler
       #         global_norm += param.grad.float().square().sum()
       # global_norm = global_norm.sqrt()
       # for param in self.optim.params: 
       #     if param.grad is not None:
       #         param.grad = (param.grad / Tensor.where(global_norm > 1.0, global_norm, 1.0)).cast(param.grad.dtype)
            
        self.optim.step()
        return loss

    @TinyJit
    def validate_step(self, input_batch: Tensor):
        Tensor.no_grad = True
        loss = self.model.criterion(input_batch)
        Tensor.no_grad = False
        return loss

    def run_epoch(self, dataloader: DataLoader, step_fn):
        epoch_loss = 0.0
        gflops = 0
        batch_time = time.perf_counter()
        for batch in dataloader:
            step_start_time = time.perf_counter()
            GlobalCounters.reset()
            loss = step_fn(batch)
            step_end_time = time.perf_counter() - step_start_time
            gflops += GlobalCounters.global_ops / (1e9 * step_end_time)
            epoch_loss += loss.float().item()
        batch_end_time = time.perf_counter() - batch_time
        gflops /= len(dataloader)
        avg_loss = epoch_loss / len(dataloader)
        return avg_loss, gflops, batch_end_time
                

    def train(self):
        print(colored(f"Starting training {self.model.__class__.__name__} with {self.epochs} epochs", 'yellow'))
 
        for epoch in range(self.epochs):
            train_loss, train_gflops, train_time = self.run_epoch(self.train_dataloader, self.train_step)
            self.train_losses.append(train_loss)
        
            val_loss, val_gflops, val_time = self.run_epoch(self.val_dataloader, self.validate_step)
            self.val_losses.append(val_loss)
        
            printing(epoch, self.epochs, train_loss, val_loss, train_gflops, val_gflops, self.optim.lr.float().item(), train_time, val_time)

            self.scheduler.step(val_loss)
        
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_model(self.model, False, True)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Best validation loss: {self.best_loss:.9f}")
        print(f"Train - Max loss: {max(self.train_losses):.9f}, Min loss: {min(self.train_losses):.9f}")
        print(f"Val - Max loss: {max(self.val_losses):.9f}, Min loss: {min(self.val_losses):.9f}")
        save_model(self.model, final=True)
        plot_loss(self.train_losses, self.val_losses, self.model)