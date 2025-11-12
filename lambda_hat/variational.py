# lambda_hat/variational.py
"""
Variational Inference utilities for Local Learning Coefficient estimation.

Implements a mixture of factor analyzers with:
- Equal means at w* (center of local posterior)
- Shared diagonal D across components for PD + Woodbury efficiency
- Rank budget r per component (low-rank + diagonal covariance)
- STL (sticking-the-landing) pathwise gradients for continuous params
- Rao-Blackwellized score gradients for mixture weights
- Optional geometry whitening via diagonal preconditioner
"""

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

# === Whitening Infrastructure ===


class Whitener(NamedTuple):
    """Geometry whitening transformation: tilde_w = A^{1/2} (w - w*)"""

    to_tilde: Callable[[jnp.ndarray], jnp.ndarray]  # w - w* -> whitened coords
    from_tilde: Callable[[jnp.ndarray], jnp.ndarray]  # whitened -> w - w*
    A_sqrt: jnp.ndarray  # (d,) diagonal matrix A^{1/2}
    A_inv_sqrt: jnp.ndarray  # (d,) diagonal matrix A^{-1/2}


def make_whitener(A_diag: Optional[jnp.ndarray], eps: float = 1e-8) -> Whitener:
    """Create geometry whitening transformation from diagonal preconditioner.

    Args:
        A_diag: SPD diagonal matrix (e.g., from Adam second moment), or None for identity
        eps: Small constant for numerical stability

    Returns:
        Whitener with forward/backward transformations
    """
    if A_diag is None:
        # Identity whitening (no transformation)
        def identity(x):
            return x

        one = jnp.array(1.0, dtype=jnp.float32)
        return Whitener(to_tilde=identity, from_tilde=identity, A_sqrt=one, A_inv_sqrt=one)

    A_diag = jnp.asarray(A_diag)
    # Clamp to avoid numerical issues
    A_diag = jnp.maximum(A_diag, eps)
    A_sqrt = jnp.sqrt(A_diag)
    A_inv_sqrt = 1.0 / A_sqrt

    return Whitener(
        to_tilde=lambda w_minus_wstar: A_sqrt * w_minus_wstar,
        from_tilde=lambda tw: A_inv_sqrt * tw,
        A_sqrt=A_sqrt,
        A_inv_sqrt=A_inv_sqrt,
    )


# === Variational Family Parameters ===


class VIParams(NamedTuple):
    """Variational parameters for mixture of factor analyzers.

    Covariance structure: Sigma_m = D + K_m K_m^T
    where K_m = D^{1/2} A_m (algebraic whitening)
    """

    rho: jnp.ndarray  # (d,) shared diagonal: D = softplus(rho)^2
    A: jnp.ndarray  # (M, d, r) low-rank factors in whitened coords
    alpha: jnp.ndarray  # (M,) mixture logits (softmax -> pi)


# === Core Variational Family Functions ===


def softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Stable softplus for enforcing positivity"""
    return jax.nn.softplus(x)


def diag_from_rho(rho: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert rho parameters to shared diagonal D.

    Args:
        rho: (d,) unconstrained parameters

    Returns:
        D_sqrt: (d,) D^{1/2} where D = softplus(rho)^2
        logdet_D: scalar log|D| = 2 * sum(log(D_sqrt))
    """
    D_sqrt = softplus(rho)  # (d,)
    logdet_D = 2.0 * jnp.sum(jnp.log(D_sqrt))
    return D_sqrt, logdet_D


def init_vi_params(key: jax.random.PRNGKey, params_flat: jnp.ndarray, M: int, r: int) -> VIParams:
    """Initialize variational parameters.

    Args:
        key: JRNG key
        params_flat: (d,) flattened initial parameters (w*)
        M: Number of mixture components
        r: Rank budget per component

    Returns:
        VIParams with small random initialization
    """
    d = params_flat.size
    dtype = params_flat.dtype

    k1, k2 = jax.random.split(key)

    # Shared diagonal: initialize to small positive values (D ≈ I after softplus)
    rho = jnp.zeros((d,), dtype=dtype)

    # Low-rank factors: small random initialization
    A = 0.05 * jax.random.normal(k1, (M, d, r), dtype=dtype)

    # Mixture logits: near-uniform (alpha ≈ 0 => pi ≈ 1/M)
    alpha = jnp.zeros((M,), dtype=dtype)

    return VIParams(rho=rho, A=A, alpha=alpha)


def sample_q(
    key: jax.random.PRNGKey,
    params: VIParams,
    wstar_flat: jnp.ndarray,
    whitener: Whitener,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Pathwise sample from variational distribution q.

    Args:
        key: JRNG key
        params: VI parameters
        wstar_flat: (d,) flattened center (w*)
        whitener: Geometry whitening transformation

    Returns:
        w_flat: (d,) sampled parameters
        aux: Dict with sampling artifacts (component m, noise z/eps, etc.)
    """
    rho, A, alpha = params
    key_m, key_z, key_eps = jax.random.split(key, 3)

    # Sample mixture component
    pi = jax.nn.softmax(alpha)
    m = jax.random.categorical(key_m, jnp.log(pi))

    # Get D^{1/2} for selected component
    D_sqrt, _ = diag_from_rho(rho)  # (d,)

    # Sample noise
    z = jax.random.normal(key_z, (A.shape[-1],))  # (r,)
    eps = jax.random.normal(key_eps, rho.shape)  # (d,)

    # Algebraic whitening: K_m = D^{1/2} A_m
    A_m = A[m]  # (d, r)
    K_m_z = (D_sqrt[:, None] * A_m) @ z  # (d,)
    D_sqrt_eps = D_sqrt * eps  # (d,)

    # Sample in whitened coordinates: tilde_v = K_m z + D^{1/2} eps
    tilde_v = K_m_z + D_sqrt_eps

    # Transform back to model coordinates: w = w* + A^{-1/2} tilde_v
    w_flat = wstar_flat + whitener.from_tilde(tilde_v)

    aux = {
        "m": m,
        "z": z,
        "eps": eps,
        "tilde_v": tilde_v,
        "D_sqrt": D_sqrt,
        "pi": pi,
    }
    return w_flat, aux


def logpdf_components_and_resp(
    tilde_v: jnp.ndarray, params: VIParams
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute per-component log-pdfs and responsibilities.

    Uses Woodbury/Sherman-Morrison identities for O(Mdr) complexity.

    Args:
        tilde_v: (d,) sample in whitened coordinates (w - w* in whitened space)
        params: VI parameters

    Returns:
        logps: (M,) per-component log N(w | w*, Sigma_m)
        r: (M,) responsibilities r_j(w) = p(component=j | w)
    """
    rho, A, alpha = params
    M, d, r = A.shape

    D_sqrt, logdet_D = diag_from_rho(rho)

    # x = D^{-1/2} v (elementwise divide)
    x = tilde_v / D_sqrt

    def one_component(A_m):
        """Compute log-pdf for one component using Woodbury identity."""
        # C = I + A^T A (r x r matrix)
        C = jnp.eye(r, dtype=A_m.dtype) + (A_m.T @ A_m)
        # Cholesky for stability
        L = jnp.linalg.cholesky(C)

        # Quadratic form: v^T Sigma^{-1} v = x^T x - x^T A (I + A^T A)^{-1} A^T x
        g = A_m.T @ x  # (r,)
        y = jax.scipy.linalg.cho_solve((L, True), g[:, None])  # (r, 1)
        quad = jnp.dot(x, x) - jnp.dot(g, y[:, 0])

        # Log-determinant: log|Sigma| = log|D| + log|I + A^T A|
        logdet = logdet_D + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        # Log-pdf: -1/2 (d log(2π) + log|Sigma| + quad)
        logp = -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet + quad)
        return logp

    # Vectorize over all components
    logps = jax.vmap(one_component)(A)  # (M,)

    # Log mixture density: log q(w) = logsumexp(log π_j + log p_j(w))
    logmix = jax.nn.logsumexp(jax.nn.log_softmax(alpha) + logps)

    # Responsibilities: r_j(w) = π_j p_j(w) / q(w) (in log space for stability)
    r = jax.nn.softmax(alpha + logps - logmix)

    return logps, r
