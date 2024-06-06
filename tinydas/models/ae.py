from typing import Optional

from tinygrad import TinyJit, dtypes, nn
from tinygrad.nn import Tensor

# from tinydas.losses import mse


def mse(X: Tensor, Y: Tensor):
    return Y.sub(X).square().mean()


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


class AE:
    def __init__(self, **kwargs):
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

    def loss_function(self, X: Tensor, Y: Tensor) -> Tensor:
        return mse(X, Y)

    def predict(self, x: Tensor) -> Tensor:
        return self(x)


if __name__ == "__main__":
    import random as rnd

    Tensor.manual_seed(574)
    rnd.seed(574)

    model = AE()
    # print(model)

    data = Tensor.ones(1, 28, 28, dtype=dtypes.float32).reshape(-1, 28 * 28)

    out = model(data)

    l = mse(data, out).item()

    print(f"Loss: {l:.3f}")

    pred = model.predict(data)

    print(f"Prediction: \n{pred.numpy().mean():.3f}")

