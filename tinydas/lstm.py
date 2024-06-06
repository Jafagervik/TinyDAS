from typing import Any, Tuple

from tinygrad import Tensor, TinyJit


class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int, dropout):
        self.dropout = dropout

        self.weights_ih = Tensor.uniform(hidden_size * 4, input_size)
        self.bias_ih = Tensor.uniform(hidden_size * 4)
        self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size)
        self.bias_hh = Tensor.uniform(hidden_size * 4)

    def __call__(self, x, hc) -> Tensor:
        gates = x.linear(self.weights_ih.T, self.bias_ih) + hc[: x.shape[0]].linear(
            self.weights_hh.T, self.bias_hh
        )

        i, f, g, o = gates.chunk(4, 1)
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

        c = (f * hc[x.shape[0] :]) + (i * g)
        h = (o * c.tanh()).dropout(self.dropout)

        return Tensor.cat(h, c).realize()


class LSTM:
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 1, dropout=0.0
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = num_layers

        self.cells = [
            (
                LSTMCell(input_size, hidden_size, dropout)
                if i == 0
                else LSTMCell(
                    hidden_size, hidden_size, dropout if i != num_layers - 1 else 0
                )
            )
            for i in range(num_layers)
        ]

    def __call__(self, x: Tensor, hc):
        @TinyJit
        def _do_step(x_, hc_):
            return self.do_step(x_, hc_)

        if hc is None:
            hc = Tensor.zeros(
                self.layers, 2 * x.shape[1], self.hidden_size, requires_grad=False
            )

        # output = Tensor.empty()
        output = None

        for t in range(x.shape[0]):
            hc = _do_step(x[t] + 1 - 1, hc)  # TODO: why do we need to do this?
            if output is None:
                output = hc[-1:, : x.shape[1]]
            else:
                output = output.cat(hc[-1:, : x.shape[1]], dim=0).realize()

        return output, hc

    def do_step(self, x, hc):
        new_hc = [x]
        for i, cell in enumerate(self.cells):
            new_hc.append(cell(new_hc[i][: x.shape[0]], hc[i]))
        return Tensor.stack(*new_hc[1:]).realize()


class Model:
    def __init__(self):
        self.l = LSTM(5, 10)

    def __call__(self, x: Tensor):
        return self.l(x, None)


def mse(X: Tensor, Y: Tensor) -> Tensor:
    return X.sub(Y).square().mean()


def test_lstm():
    Tensor.manual_seed(1234)
    # lstm = LSTM(10, 20, 2)
    x = Tensor.uniform(5, 5)

    model = Model()

    out, hc = model(x)
    print(out.shape)
    print(out.numpy())


if __name__ == "__main__":
    test_lstm()
