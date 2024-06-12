from tinygrad.nn import Tensor

# This need to be backwarded after used in te forward pass


def mse(X: Tensor, Y: Tensor) -> Tensor:
    return Y.sub(X).square().mean()


def mae(X: Tensor, Y: Tensor) -> Tensor:
    return Y.sub(X).abs().mean()


def kl_divergence(mu: Tensor, logvar: Tensor):
    return 0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(axis=1).mean()


def reconstruct(mu: Tensor, logvar: Tensor):
    pass


def elbo_loss(encoder, decoder, X: Tensor):
    mu, logvar = encoder(X)

    z = reconstruct(mu, logvar)

    decoded = decoder(z)

    kl = kl_div(X, decoded)
    recloss = mse(X, decoded)

    elbo = kl + recloss

    return {"elbo": elbo, "kl": kl, "rec": recloss}
