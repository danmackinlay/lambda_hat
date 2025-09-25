import jax.numpy as jnp
from jax import value_and_grad, grad, jit
from .config import Config

def compute_beta_gamma(cfg: Config, d: int) -> tuple[float, float]:
    n = max(3, int(cfg.n_data))
    beta = cfg.beta0 / jnp.log(n) if cfg.beta_mode == "1_over_log_n" else cfg.beta0
    gamma = (d/(cfg.prior_radius**2)) if (cfg.prior_radius is not None) else cfg.gamma
    return float(beta), float(gamma)

def make_logpost_and_score(loss_full, loss_minibatch, theta0, n, beta, gamma):
    beta, gamma, n = map(lambda x: jnp.asarray(x, theta0.dtype), (beta, gamma, n))
    def logpost(theta):
        Ln = loss_full(theta)
        lp = -0.5 * gamma * jnp.sum((theta - theta0)**2)
        return lp - n * beta * Ln
    logpost_and_grad = value_and_grad(logpost)
    @jit
    def grad_logpost_minibatch(theta, minibatch):
        Xb, Yb = minibatch
        gLb = grad(lambda th: loss_minibatch(th, Xb, Yb))(theta)
        return -gamma*(theta-theta0) - beta*n*gLb
    return logpost_and_grad, grad_logpost_minibatch

def make_logdensity_for_mclmc(loss_full64, theta0_f64, n, beta, gamma):
    beta, gamma, n = map(lambda x: jnp.asarray(x, theta0_f64.dtype), (beta, gamma, n))
    @jit
    def logdensity(theta64):
        Ln = loss_full64(theta64)
        lp = -0.5 * gamma * jnp.sum((theta64 - theta0_f64)**2)
        return lp - n * beta * Ln
    return logdensity