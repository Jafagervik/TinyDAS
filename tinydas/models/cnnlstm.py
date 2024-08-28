from typing import Dict, Tuple, List, Callable

from tinydas.convblocks import ConvBlock, DeconvBlock
from tinydas.lstm import LSTM
from tinygrad import TinyJit, Tensor
from tinygrad.nn import Conv2d, Linear, ConvTranspose2d, BatchNorm

from tinydas.losses import mse
from tinydas.models.base import BaseAE

class CNNLSTMAE(BaseAE):
    def __init__(self, input_shape=(625, 2137), hidden_sizes=[16, 32], lstm_hidden=512, lstm_layers=1, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.encoder: List[Callable[[Tensor], Tensor]] = [
            Conv2d(1, hidden_sizes[0], 3, stride=2, padding=1),
            lambda x: x.relu(),
            Conv2d(hidden_sizes[0], hidden_sizes[1], 3, stride=2, padding=1),
            lambda x: x.relu(),
        ]

        test_input = Tensor.zeros(1, 1, *input_shape)
        for layer in self.encoder:
            test_input = layer(test_input)
        
        self.encoder_output_shape = test_input.shape[1:]
        
        self.encoder_lstm = LSTM(self.encoder_output_shape[0], lstm_hidden, lstm_layers)

        self.decoder: List[Callable[[Tensor], Tensor]] = [
            Linear(lstm_hidden, self.encoder_output_shape[0]),
            lambda x: x.relu(),
            ConvTranspose2d(hidden_sizes[1], hidden_sizes[0], 3, stride=2, padding=1, output_padding=1),
            lambda x: x.relu(),
            ConvTranspose2d(hidden_sizes[0], 1, 3, stride=2, padding=1, output_padding=1),
            lambda x: x.sigmoid()
        ]

    def encode(self, x: Tensor) -> Tensor:
        x = x.reshape(shape=(-1, 1, *self.input_shape))
        for layer in self.encoder:
            x = layer(x)
        
        batch_size, channels, height, width = x.shape
        x = x.permute(2, 3, 0, 1).reshape(height * width, batch_size, channels)
        x, _ = self.encoder_lstm(x)
        return x

    def decode(self, x: Tensor) -> Tensor:
        height, width = self.encoder_output_shape[1:]
        x = x.reshape(height, width, x.shape[1], -1).permute(2, 3, 0, 1)
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return (decoded,)

    def criterion(self, x: Tensor) -> Tensor:
        (y,) = self(x)
        return mse(x, y)

    @staticmethod
    def loss(out: Tensor, pred: Tensor) -> Tensor:
        return ((out - pred) ** 2).mean()

    def predict(self, x: Tensor) -> Tensor:
        x = x.reshape(1, 1, *self.input_shape)
        (out,) = self(x)
        out = out.squeeze()
        return out.realize()