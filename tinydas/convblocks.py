from tinygrad.nn import BatchNorm, Conv2d, ConvTranspose2d, Tensor


class ConvBlock:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        self.conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def __call__(self, x: Tensor) -> Tensor:
        return self.conv(x).relu().max_pool2d(kernel_size=2,stride=2)

class DeconvBlock:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=2, output_padding=1):
        self.deconv = ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding)

    def __call__(self, x: Tensor) -> Tensor:
        return self.deconv(x).relu()