from typing import Optional

from tinygrad.nn import Linear, Tensor


class LinearBlockLayer:
    def __init__(self, i: int, o: int, do: Optional[float] = None):
        self.net = [
            Linear(i, o),
            # Tensor.batchnorm,
            Tensor.relu,
        ]
        if do:
            self.net.append(Tensor.dropout)

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)
