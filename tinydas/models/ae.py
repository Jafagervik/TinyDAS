from typing import Optional

from tinygrad import TinyJit, dtypes, nn
from tinygrad.nn import Tensor

from tinydas.losses import mse
from tinydas.models.base import BaseAE


class LL:
    def __init__(self, i: int, o: int, do: Optional[float] = None) -> None:
        self.net = [
            nn.Linear(i, o),
            Tensor.relu,
        ]
        if do:
            self.net.append(Tensor.dropout)

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class AE(BaseAE):
    def __init__(self, **kwargs):
        super().__init__()
        self.M = 28
        self.N = 28
        self.inp = self.M * self.N
        self.hidden = 256
        self.latent = 64

        self.net = [
            LL(self.inp, self.hidden, do=0.2),
            LL(self.hidden, self.latent),
            LL(self.latent, self.hidden),
            nn.Linear(self.hidden, self.inp),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)

    def criterion(self, X: Tensor) -> Tensor:
        return mse(X, model(X))


if __name__ == "__main__":
    model = AE()

    data = Tensor.ones(1, 28, 28, dtype=dtypes.float32).reshape(-1, 28 * 28)

    out = model(data)

    l = mse(data, out).item()

    print(f"Loss: {l:.3f}")

    pred = model.predict(data)

    print(f"Prediction: \n{pred.numpy().mean():.3f}")
