from tinygrad.nn import Tensor


def mse(X: Tensor, Y: Tensor) -> Tensor:
    return Y.sub(X).square().mean()


def mae(X: Tensor, Y: Tensor) -> Tensor:
    return Y.sub(X).abs().mean()


def kl_divergence(mu: Tensor, logvar: Tensor):
    """
    Compute the KL divergence between a normal distribution with mean mu and logvar and a standard normal distribution.
    """
    return -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(axis=1).mean()
