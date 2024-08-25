from tinygrad.nn import BatchNorm, Conv2d, ConvTranspose2d, Tensor


class ConvBlock:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        act=lambda x: x.relu(),
        bn: bool = True
    ) -> None:
        self.c = Conv2d(
                in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        self.b = BatchNorm(out_channels)
        #self.b.weight.requires_grad = False
        self.bn = bn
        self.act = act

    def __call__(self, x: Tensor) -> Tensor:
        x = self.c(x)
        if self.bn:
            x = self.b(x)
        return self.act(x)


class DeconvBlock:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        act = lambda x: x.relu(),
        bn: bool = True
    ) -> None:
        self.c = ConvTranspose2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.b = BatchNorm(out_channels)
        #self.b.weight.requires_grad = False
        self.act = act

    def __call__(self, x: Tensor) -> Tensor:
        x = self.c(x)
        x = self.b(x)
        return self.act(x)
