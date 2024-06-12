import random
from typing import Tuple

import matplotlib

from tinydas.losses import kl_divergence, mae
from tinydas.models.base import BaseAE

matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt
from tinygrad import TinyJit, nn
from tinygrad.nn import Tensor
from tqdm import trange

from tinydas.utils import reparameterize


class LinearLayer:
    def __init__(self, in_features: int, out_features: int):
        self.net = [
            nn.Linear(in_features, out_features),
            Tensor.relu,
            Tensor.dropout,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class Encoder:
    def __init__(self):
        self.layers = [
            LinearLayer(28 * 28, 512),
            LinearLayer(512, 256),
            LinearLayer(256, 64),
        ]

        self.mu = nn.Linear(64, 32)
        self.logvar = nn.Linear(64, 32)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.sequential(self.layers)
        return self.mu(x), self.logvar(x)


class Decoder:
    def __init__(self):
        self.layers = [
            LinearLayer(32, 64),
            LinearLayer(64, 256),
            LinearLayer(256, 512),
            nn.Linear(512, 28 * 28),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


class VAE(BaseAE):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x: Tensor) -> Tensor:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z)

    def criterion(self, x: Tensor) -> Tensor:
        mu, logvar = self.encoder(x)
        x_hat = self(x)
        rec_loss = mae(x, x_hat)
        kl_loss = kl_divergence(mu, logvar)
        elbo_loss = rec_loss + kl_loss
        return elbo_loss
    
    def predict(self, x: Tensor) -> Tensor:
        return self(x)

    def plot_latent_space(self, x: Tensor):
        mu, _ = self.encoder(x)
        plt.scatter(
            mu[:, 0].numpy(), mu[:, 1].numpy(), c=range(x.shape[0]), cmap="viridis"
        )
        plt.show()


def plot_losses(losses):
    transposed_data = list(zip(*losses))
    x = range(1, len(losses) + 1)
    names = ["ELBO", "KL", "Rec"]

    for i, y in enumerate(transposed_data):
        plt.plot(x, y, label=f"{names[i]}")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Loss")
    plt.legend()
    plt.show()


def train():
    Tensor.manual_seed(0)
    random.seed(0)
    M, N = 28, 28

    model = VAE()
    optim = nn.optim.Adam(nn.state.get_parameters(model))

    X = Tensor.randn((32, M * N))
    EPS = 30
    bs = 4
    losses = [[0.0, 0.0, 0.0]] * EPS

    @TinyJit
    def step() -> Tuple[Tensor, Tensor, Tensor]:
        with Tensor.train():
            samples = Tensor.randint(bs, high=X.shape[0])
            x = X[samples].reshape(-1, M * N)
            optim.zero_grad()
            x_hat, mu, logvar = model(x)
            elbo, rec, kl = model.criterion(x, x_hat, mu, logvar)
            elbo.backward()
            optim.step()
            return elbo, rec, kl

    print("Training VAE...")

    for epoch in (t := trange(EPS)):
        elbo, kl, rec = step()
        losses[epoch] = [elbo.item(), kl.item(), rec.item()]
        t.set_description(f"Epoch {epoch+1} | Loss: {elbo.item():.2f}")

    plot_losses(losses)


if __name__ == "__main__":
    train()
