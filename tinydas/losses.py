from tinygrad.nn import Tensor


def mse(X: Tensor, Y: Tensor) -> Tensor:
    return Y.sub(X).square().mean()


def mae(X: Tensor, Y: Tensor) -> Tensor:
    return Y.sub(X).abs().mean()

def huber_loss(y_hat: Tensor, y: Tensor, delta: float) -> Tensor:
    abs_err = y_hat.sub(y).abs()
    quad = 0.5 * abs_err.square()
    linear = delta * abs_err - 0.5 * (delta ** 2)
    loss = Tensor.where(abs_err <= delta, quad, linear)
    return loss.mean().realize()

def hinge_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    return Tensor.maximum(0, 1-y_hat*y).mean().realize()


def kl_divergence(mu: Tensor, logvar: Tensor):
    """
    Compute the KL divergence between a normal distribution with mean mu and logvar and a standard normal distribution.
    """
    return -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(axis=1).mean()
