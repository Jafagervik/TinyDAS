from typing import Dict, List, Tuple

from tinygrad import nn, TinyJit
from tinygrad.nn import Tensor

from tinydas.linearblock import LinearBlockLayer
from tinydas.losses import kl_divergence, mse
from tinydas.models.base import BaseAE
from tinydas.utils import reparameterize, minmax


class Encoder:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        do: float,
    ):
        self.net = [
            LinearBlockLayer(input_dim, hidden_dims[0], do),
        ]
        for i in range(len(hidden_dims) - 1):
            self.net.append(LinearBlockLayer(hidden_dims[i], hidden_dims[i + 1], do))

        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.sequential(self.net)
        return self.mu(x), self.logvar(x)


class Decoder:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        do: float,
    ):
        self.net = [
            LinearBlockLayer(latent_dim, hidden_dims[0], do),
        ]
        for i in range(len(hidden_dims) - 1):
            self.net.append(LinearBlockLayer(hidden_dims[i], hidden_dims[i + 1], do))

        self.eps = 1e-8
        self.last = nn.Linear(hidden_dims[-1], input_dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.sequential(self.net)
        return self.last(x).sigmoid().clip(self.eps, 1-self.eps)



class VAE(BaseAE):
    def __init__(self, devices: List[str], **kwargs):
        super().__init__()

        hidden_layers = kwargs["mod"]["hidden_layers"]

        self.encoder = Encoder(
            kwargs["mod"]["M"] * kwargs["mod"]["N"],
            hidden_layers,
            kwargs["mod"]["latent"],
            kwargs["mod"]["p"],
        )

        self.kld_weight = kwargs["mod"]["kld_weight"]

        hidden_layers = hidden_layers[::-1]

        self.decoder = Decoder(
            kwargs["mod"]["M"] * kwargs["mod"]["N"],
            hidden_layers,
            kwargs["mod"]["latent"],
            kwargs["mod"]["p"],
        )

    @property
    def convolutional(self) -> bool:
        return False

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        x_hat, mu, logvar = self(x)

        print(x_hat)
        print(mu)
        print(logvar)

        rec_loss = mse(x, x_hat)

        kl_loss = kl_divergence(mu, logvar)

        return {"loss": kl_loss, "klloss": rec_loss, "recloss": rec_loss}


    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        """
        Input tensor is being processed to fit encoder
        after decoder is done, it is reshaped back
        """
        Tensor.no_grad = True
        x = x.reshape(1, 625 * 2137)
        x = minmax(x)
        (out, _, _) = self(x)

        out = out.reshape(625, 2137)
        return out.realize()

#    def plot_latent_space(self, x: Tensor):
#        mu, _ = self.encoder(x)
#        plt.scatter(
#            mu[:, 0].numpy(), mu[:, 1].numpy(), c=range(x.shape[0]), cmap="viridis"
#        )
#        plt.show()


# def plot_losses(losses):
#    transposed_data = list(zip(*losses))
#    x = range(1, len(losses) + 1)
#    names = ["ELBO", "KL", "Rec"]
#
#    for i, y in enumerate(transposed_data):
#        plt.plot(x, y, label=f"{names[i]}")
#
#    plt.xlabel("Epoch")
#    plt.ylabel("Value")
#    plt.title("Loss")
#    plt.legend()
#    plt.show()
