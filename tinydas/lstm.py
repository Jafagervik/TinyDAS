from typing import Callable, List, Optional, Tuple

from tinygrad import GlobalCounters, Tensor, TinyJit, nn
from tqdm import trange


class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int, dropout=0.0):
        self.dropout = dropout

        self.weights_ih = Tensor.uniform(hidden_size * 4, input_size)
        self.bias_ih = Tensor.uniform(hidden_size * 4)
        self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size)
        self.bias_hh = Tensor.uniform(hidden_size * 4)

    def __call__(self, x: Tensor, hc: Tensor) -> Tensor:
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

    def __call__(self, x: Tensor, hc: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        @TinyJit
        def _do_step(x_: Tensor, hc_: Tensor) -> Tensor:
            return self.do_step(x_, hc_)

        if hc is None:
            hc = Tensor.zeros(
                self.layers, 2 * x.shape[1], self.hidden_size, requires_grad=False
            )

        output = None

        for t in range(x.shape[0]):
            hc = _do_step(x[t] + 1 - 1, hc)  # TODO: why do we need to do this?
            output = (
                hc[-1:, : x.shape[1]]
                if output is None
                else output.cat(hc[-1:, : x.shape[1]], dim=0)
            )
            # if output is None:
            #     output = hc[-1:, : x.shape[1]]
            # else:
            #     output = output.cat(hc[-1:, : x.shape[1]], dim=0).realize()

        return output, hc

    def do_step(self, x: Tensor, hc: Tensor) -> Tensor:
        new_hc = [x]
        for i, cell in enumerate(self.cells):
            new_hc.append(cell(new_hc[i][: x.shape[0]], hc[i]))
        return Tensor.stack(*new_hc[1:]).realize()


class Model:
    def __init__(self):
        self.l = LSTM(5, 1)

    def __call__(self, x: Tensor) -> Tensor:
        out, hc = self.l(x)
        return out.relu()


if __name__ == "__main__":
    Tensor.manual_seed(42069)
    model = Model()
    lr = 2.0

    params = nn.state.get_parameters(model)
    opt = nn.optim.Adam(params, lr)

    data = Tensor.ones(10, 5, 5)

    # @TinyJit
    def train_step() -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            loss = model(data).sub(data).square().mean().backward()
            opt.step()
            return loss

    for i in (t := trange(5)):
        GlobalCounters.reset()
        loss = train_step()
        t.set_description(f"Epoch {i+1} <> Loss: {loss.item():.4f}")

    newd = Tensor.ones(5, 5)

    out = model(newd).reshape(5, 5)

    print(out.numpy())
