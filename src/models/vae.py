from typing import Tuple

from tinygrad import nn
from tinygrad.nn import Tensor


class Encoder:
    def __init__(self):
        self.layers = [
            nn.Linear(2137 * 7500, 16384),
            Tensor.gelu,
            Tensor.dropout,
            nn.Linear(16384, 4096),
            Tensor.gelu,
            Tensor.dropout,
            nn.Linear(4096, 1024),
            Tensor.gelu,
            Tensor.dropout,
        ]
        self.mu = nn.Linear(1024, 256)
        self.logvar = nn.Linear(1024, 256)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.sequential(self.layers)
        return self.mu(x), self.logvar(x)


class Decoder:
    def __init__(self):
        self.layers = [
            nn.Linear(256, 1024),
            Tensor.gelu,
            Tensor.dropout,
            nn.Linear(1024, 4096),
            Tensor.gelu,
            Tensor.dropout,
            nn.Linear(4096, 16384),
            Tensor.gelu,
            Tensor.dropout,
            nn.Linear(16384, 2137 * 7500),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


def reconstruct(mu: Tensor, logvar: Tensor) -> Tensor:
    std = logvar.exp().sqrt()
    eps = Tensor.randn(mu.shape)
    return mu + eps * std


def elbo_loss(x: Tensor, x_hat: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    rec_loss = x.sub(x_hat).square().mean()
    kl_loss = 0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(axis=1).mean()
    return rec_loss + kl_loss


class VAE:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x: Tensor):
        mu, logvar = self.encoder(x)
        z = reconstruct(mu, logvar)
        return self.decoder(z), mu, logvar


def infer():
    model = VAE()
    x = Tensor.randn((32, 2137 * 7500))
    x_hat, mu, logvar = model(x)
    print(x_hat.shape, mu.shape, logvar.shape)


if __name__ == "__main__":
    infer()

