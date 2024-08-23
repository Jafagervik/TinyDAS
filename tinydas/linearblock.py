from tinygrad.nn import Linear, Tensor


class LinearBlockLayer:
    def __init__(self, i: int, o: int, do: float = 0.0):
        self.l = Linear(i, o)
        self.dropout = do

    def __call__(self, x: Tensor) -> Tensor:
        return self.l.leakyrelu(0.2).dropout(self.dropout)
