from tinygrad.nn import BatchNorm2d, Conv2d, ConvTranspose2d, Tensor


class ConvBlock:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        act=Tensor.leakyrelu,
    ) -> None:
        self.net = [
            Conv2d(
                in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(out_channels),
            act,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)


class DeconvBlock:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        act=Tensor.leakyrelu,
    ) -> None:
        self.net = [
            ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            BatchNorm2d(out_channels),
            act,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)
