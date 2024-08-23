from typing import Optional
from tinydas.kl import AdaptiveKLWeight
from tinygrad.nn import Tensor

def mse(X: Tensor, Y: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == 'mean': return ((X - Y)**2).mean()
    if reduction == 'sum': return ((X - Y)**2).sum()
    if reduction == 'none': return ((X - Y) ** 2)

def mae(X: Tensor, Y: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == 'mean': return ((X - Y).abs()).mean()
    if reduction == 'sum': return ((X - Y).abs()).sum()
    if reduction == 'none': return ((X - Y).abs())
        

def huber_loss(y_hat: Tensor, y: Tensor, delta: float) -> Tensor:
    abs_err = y_hat.sub(y).abs()
    quad = 0.5 * abs_err.square()
    linear = delta * abs_err - 0.5 * (delta ** 2)
    loss = Tensor.where(abs_err <= delta, quad, linear)
    return loss.mean().realize()

def hinge_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    return Tensor.maximum(0, 1-y_hat*y).mean().realize()

def ssim_loss(y: Tensor, y_hat: Tensor) -> Tensor :
    mu_y = y.mean()
    mu_y_hat = y_hat.mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_y_hat = ((y_hat - mu_y_hat) ** 2).mean()
    sigma_y_y_hat = ((y - mu_y) * (y_hat - mu_y_hat)).mean()
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu_y * mu_y_hat + c1) * (2 * sigma_y_y_hat + c2)) / ((mu_y ** 2 + mu_y_hat ** 2 + c1) * (sigma_y + sigma_y_hat + c2))
    return (1 - ssim.mean()).realize()

def log_likelihood_loss(y: Tensor, y_hat: Tensor) -> Tensor:
    return -(y * y_hat.log() + (1 - y) * (1 - y_hat).log()).sum().realize()

def total_correlation_loss(z: Tensor, q_z: Tensor) -> Tensor:
    log_q_z_product = (q_z.log()).sum(axis=-1)
    log_q_z = q_z.mean(axis=0).log()
    return (log_q_z - log_q_z_product).mean().realize()

def cross_entropy(x:Tensor, y:Tensor, reduction:str='mean', label_smoothing:float=0.0) -> Tensor:
    divisor = y.shape[1]
    assert isinstance(divisor, int), "only supported int divisor"
    y = (1 - label_smoothing)*y + label_smoothing / divisor
    ret = -x.log_softmax(axis=1).mul(y).sum(axis=1)
    if reduction=='none': return ret
    if reduction=='sum': return ret.sum()
    if reduction=='mean': return ret.mean()
    raise NotImplementedError(reduction)


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    return (0.5 * (logvar.exp().clip(min_=1e-5, max_=1e3) + mu ** 2 - 1 - logvar).sum(axis=1)).sum(axis=0)
    #return -0.5 * Tensor.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return -0.5 * Tensor.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1).mean()
    #return -0.5 * Tensor.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1).mean()
    #var = logvar.softplus() ** 2
    #kl_div = -0.5 * (mu ** 2 + var - 1 - logvar).sum(axis=1)
    #return kl_div.mean(axis=0)


def elbo(x: Tensor, y: Tensor, mu: Tensor, logvar: Tensor, kl_weight: float=0.0):
    recon_loss = mse(x, y, reduction='mean')
    kld_loss = kl_divergence(mu, logvar)
    
    return recon_loss + kl_weight * kld_loss, recon_loss, kld_loss