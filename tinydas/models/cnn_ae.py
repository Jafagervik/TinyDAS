from typing import Any
from tinygrad.nn import Tensor
from tinygrad import nn
from tinydas.losses import mse

class Encoder:
    def __init__(self) -> None:
        self.c1 = self.conv_block(1, 32)
        self.c2 = self.conv_block(32, 64)
        self.c3 = self.conv_block(64, 128)
        self.fc = nn.Linear(1000, 128)

    def conv_block(self, inp: int, outp: int):
        return nn.Conv2d(in_channels=inp, out_channels=outp, kernel_size=(3,3), stride=2, padding=1)

    def __call__(self, X: Tensor): 
        X = self.c1(X)
        X = self.c2(X)
        X = self.c2(X)
        X = X.flatten()
        return self.fc(X)

class Decoder:
    def __init__(self):
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        # store the shape before flattening
        self.reshape_dim = shape_before_flattening
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # define final convolutional layer to generate output image
        self.conv1 = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), *self.reshape_dim)
        x = nn.relu(self.deconv1(x))
        x = nn.relu(self.deconv2(x))
        x = nn.relu(self.deconv3(x))
        x = nn.sigmoid(self.conv1(x))
        return x

