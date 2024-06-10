"""Beta variational autoencoder model."""

from tinygrad import Tensor


class BetaVAE:
    def __init__(self, input_shape, latent_dim, beta=1.0):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()

    def __call__(self, x: Tensor) -> Tensor:
        return self.vae(x)
