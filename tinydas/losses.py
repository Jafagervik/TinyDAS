from tinygrad.nn import Tensor


def mse(model, X: Tensor):
    return model(X).sub(X).square().mean()


def mae(model, X: Tensor):
    return model(X).sub(X).abs().mean()


def kl_div(X: Tensor, Y: Tensor):
    pass


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

