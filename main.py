from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad.nn.optim import LARS, Adam
from tinygrad import dtypes, TinyJit, Device, GlobalCounters
from tinygrad.helpers import getenv
from typing import List
from tinydas.losses import mse
from tinydas.utils import *
from tinydas.dataset import Dataset
from tinydas.dataloader import DataLoader, create_data_loaders
from tinydas.lr_schedule import ReduceLROnPlateau
from tinydas.anomalies import predict_file
from tinydas.plots import plot_das_as_heatmap, plot_loss
from tinydas.early_stopping import EarlyStopping
from tinydas.trainer import Trainer
from tinydas.selections import Opti, select_model, select_optimizer

def get_data(devices: List[str], **config):
    dataset = Dataset(path=config["data"]["dataset"], n=config["data"]["nfiles"])
    train_loader, val_loader = create_data_loaders(
        dataset, 
        batch_size=config["data"]["batch_size"],
        devices=devices, 
        num_workers=config["data"]["num_workers"],
        shuffle=config["data"]["shuffle"],
    )
    return train_loader, val_loader

def train_mode(args):
    config = get_config(args.model)
    seed_all(config["data"]["seed"])

    dtypes.default_float = dtypes.half if config["data"]["half_prec"] else dtypes.float32

    devices = get_gpus(args.gpus) 

    tl, vl = get_data(devices, **config)

    model = select_model(args.model, **config)
    # Load was here before

    if config["data"]["half_prec"]: model.half()
    if args.load: load_model(model)
    print("Data type", model.dtype)

    if len(devices) > 1: model.send_copy(devices)

    optim = select_optimizer(Opti.ADAM, model.parameters(), **config["opt"])
    scheduler = ReduceLROnPlateau(optim, patience=config["opt"]["patience"], factor=config["opt"]["factor"])

    trainer = Trainer(model, tl, vl, optim, scheduler, **config)
    trainer.train()

    #plot_loss(trainer.losses, trainer.model, save=True)


def show_imgs(args, devices: List[str], filename: str = ""):
    config = get_config(args.model)
    model = select_model(args.model, devices, **config)
    load_model(model)

    filename = filename or "./infer/20190415_032000.hdf5"
    data = load_das_file_no_time(filename)
    data = minmax(data).cast(dtypes.float16)

    filename = format_filename(filename)

    plot_das_as_heatmap(
        data.numpy(), filename, show=True, path=f"figs/{args.model}/before/{filename}.png"
    )

    reconstructed = model.predict(data)

    plot_das_as_heatmap(
        reconstructed.numpy(), filename, show=True, path=f"figs/{args.model}/after/{filename}.png"
    )

    return data, reconstructed

def find_recons(args, devices: List[str], filename: str = ""):
    config = get_config(args.model)
    model = select_model(args.model, devices, **config)
    load_model(model)

    data = load_das_file_no_time(filename)
    data = minmax(data).cast(dtypes.float16)

    filename = format_filename(filename)

    reconstructed = model.predict(data)

    return data, reconstructed

def anomaly_mode(args):
    # stream or img mode

    config = get_config(args.model)
    filename = "/cluster/home/jorgenaf/TinyDAS/data/20200302_081015.hdf5"

    predict_file(filename, args.model, **config)

    # img mode

    # das_img = None

    # model = load_model()

    # anomalies = find_anomalies(das_img, model)

    # plot anomalies

    # find anomaly scores and so on


def main():
    args = parse_args()

    match args.type:
        case "train":
            train_mode(args)
        case "detect":
            anomaly_mode(args)
        case "show":
            show_imgs(args)
        case _:
            print("Invalid mode, please select train or detect.")
            exit(1)


if __name__ == "__main__":
    main()

"""
devices = get_gpus(4)
print(f"Training on {devices}")
load = False

class AE:
    def __init__(self, input_shape=(625, 2137), latent_dim=128):
        self.input_shape = input_shape
        self.flattened_dim = input_shape[0] * input_shape[1]
        
        self.encoder1 = Linear(self.flattened_dim, 512, bias=True)
        self.encoder2 = Linear(512, latent_dim, bias=True)
        
        self.decoder1 = Linear(latent_dim, 512, bias=True)
        self.decoder2 = Linear(512, self.flattened_dim, bias=True)

    def encode(self, x):
        x = x.reshape(shape=(-1, self.flattened_dim))
        x = self.encoder1(x).relu()
        x = self.encoder2(x)
        return x

    def decode(self, x):
        x = self.decoder1(x).relu()
        x = self.decoder2(x)
        x = x.reshape(shape=(-1, *self.input_shape))
        return x

    def __call__(self, x):
        return self.decode(self.encode(x))

input_shape = (625, 2137)
model = AE(input_shape=input_shape)
if load: load_model(model)

for param in get_state_dict(model).values():
    param.replace(param.cast(dtypes.default_float))

for k, x in get_state_dict(model).items():
    x.to_(devices)

batch_size = 256
base_learning_rate = 0.1
optimizer = Adam(get_parameters(model), lr=base_learning_rate)
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
loss_scaler = 128.0 if dtypes.default_float == dtypes.float16 else 1.0
train_loader, val_loader = get_data(devices)

@TinyJit
@Tensor.train()
def train_step(input_batch: Tensor):
    optimizer.zero_grad()
    output = model(input_batch)
    loss = mse(output, input_batch)
    (loss * loss_scaler).backward()

    global_norm = Tensor([0.0], dtype=dtypes.float32, device=optimizer.params[0].device).realize()
    for param in optimizer.params: 
        if param.grad is not None:
            param.grad = param.grad / loss_scaler
            global_norm += param.grad.float().square().sum()
    global_norm = global_norm.sqrt()
    for param in optimizer.params: 
        if param.grad is not None:
            param.grad = (param.grad / Tensor.where(global_norm > 1.0, global_norm, 1.0)).cast(param.grad.dtype)
            
    optimizer.step()
    return loss

@TinyJit
def validate_step(input_batch: Tensor):
    Tensor.no_grad = True
    output = model(input_batch)
    loss = mse(output, input_batch)
    Tensor.no_grad = False
    return loss

num_epochs = 100
best_loss = float('inf')
train_losses = []
val_losses = []

def run_epoch(data_loader, step_function):
    epoch_loss = 0.0
    gflops = 0
    batch_time = time.perf_counter()
    for batch in data_loader:
        step_start_time = time.perf_counter()
        GlobalCounters.reset()
        loss = step_function(batch)
        step_end_time = time.perf_counter() - step_start_time
        gflops += GlobalCounters.global_ops / (1e9 * step_end_time)
        epoch_loss += loss.float().item()
    batch_end_time = time.perf_counter() - batch_time
    gflops /= batch_size 
    avg_loss = epoch_loss / batch_size
    return avg_loss, gflops, batch_end_time

print("START")
for epoch in range(num_epochs):
    train_loss, train_gflops, train_time = run_epoch(train_loader, train_step)
    train_losses.append(train_loss)
    
    val_loss, val_gflops, val_time = run_epoch(val_loader, validate_step)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.5f}, Train GFLOPS: {train_gflops:7.0f}, "
          f"Val Loss: {val_loss:.5f}, Val GFLOPS: {val_gflops:7.0f}, "
          f"LR: {optimizer.lr.float().item():.4f}, Time: {train_time:.2f}s")

    if val_loss < best_loss:
        best_loss = val_loss
        save_model(model, False, True)
    
    scheduler.step(val_loss)
    print(f"Learning Rate: {optimizer.lr.float().item():.4f}")
"""

