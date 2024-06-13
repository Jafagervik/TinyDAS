from typing import Tuple, Dict

import matplotlib

from tinydas.losses import kl_divergence, mae, mse
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
    def __init__(self, i: int, h: int, l: int):
        self.layers = [
            LinearLayer(625 * 2137, 128),
            LinearLayer(128, 32),
        ]

        self.mu = nn.Linear(32, 4)
        self.logvar = nn.Linear(32, 4)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.sequential(self.layers)
        return self.mu(x), self.logvar(x)


class Decoder:
    def __init__(self, i: int, h: int, l: int):
        self.layers = [
            LinearLayer(l, h),
            nn.Linear(h, i), Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


class VAE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder(
            kwargs["M"] * kwargs["N"], kwargs["hidden"], kwargs["latent"]
        )
        self.decoder = Decoder(
            kwargs["M"] * kwargs["N"], kwargs["hidden"], kwargs["latent"]
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        x_hat, mu, logvar = self(x)

        rec_loss = mse(x, x_hat)
        kl_loss = kl_divergence(mu, logvar)

        elbo_loss = rec_loss + kl_loss
        return {
            "loss": elbo_loss,
            "klloss": kl_loss,
            "recloss": rec_loss
        }
    
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
