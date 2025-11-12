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
import optax

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


def diag_from_rho(rho: jnp.ndarray, eps: float = 1e-8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert rho parameters to shared diagonal D with stability clipping.

    Args:
        rho: (d,) unconstrained parameters
        eps: Small constant for numerical stability

    Returns:
        D_sqrt: (d,) D^{1/2} where D = softplus(rho)^2, clipped to [1e-4, 1e2]
        logdet_D: scalar log|D| = 2 * sum(log(D_sqrt))
    """
    D_sqrt = softplus(rho)  # (d,)
    # Clip to safe range for float32 stability
    D_sqrt = jnp.clip(D_sqrt, 1e-4, 1e2)
    logdet_D = 2.0 * jnp.sum(jnp.log(D_sqrt + eps))  # Add eps for log stability
    return D_sqrt, logdet_D


def normalize_columns(A: jnp.ndarray, max_norm: float = 10.0) -> jnp.ndarray:
    """Normalize columns of A to prevent explosion (stability for float32).

    Args:
        A: (d, r) matrix
        max_norm: Maximum allowed column norm

    Returns:
        A with columns clipped to max_norm
    """
    norms = jnp.linalg.norm(A, axis=0, keepdims=True)  # (1, r)
    scale = jnp.minimum(1.0, max_norm / (norms + 1e-8))
    return A * scale


def init_vi_params(key: jax.random.PRNGKey, params_flat: jnp.ndarray, M: int, r: int) -> VIParams:
    """Initialize variational parameters with stability features.

    Args:
        key: JRNG key
        params_flat: (d,) flattened initial parameters (w*)
        M: Number of mixture components
        r: Rank budget per component

    Returns:
        VIParams with small random initialization and column normalization
    """
    d = params_flat.size
    dtype = params_flat.dtype

    k1, k2 = jax.random.split(key)

    # Shared diagonal: initialize to small positive values (D ≈ I after softplus)
    rho = jnp.zeros((d,), dtype=dtype)

    # Low-rank factors: small random initialization with column normalization
    A_raw = 0.05 * jax.random.normal(k1, (M, d, r), dtype=dtype)
    # Normalize columns of each component for stability
    A = jax.vmap(normalize_columns)(A_raw)

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
        # C = I + A^T A + eps*I (r x r matrix, ridge for float32 stability)
        eps_ridge = 1e-6
        C = jnp.eye(r, dtype=A_m.dtype) * (1.0 + eps_ridge) + (A_m.T @ A_m)
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


# === Optimizer State ===


class VIOptState(NamedTuple):
    """Optimizer state for VI training."""

    opt_state: optax.OptState  # Optax optimizer state
    baseline: jnp.ndarray  # Scalar EMA baseline for RB estimator
    step: jnp.ndarray  # int32 step counter


# === ELBO Optimization ===


def build_elbo_step(
    loss_batch_fn: Callable,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    wstar_flat: jnp.ndarray,
    n_data: int,
    beta: float,
    gamma: float,
    batch_size: int,
    whitener: Whitener,
    optimizer: optax.GradientTransformation,
) -> Callable:
    """Build one ELBO optimization step with STL and RB gradients.

    Args:
        loss_batch_fn: Function (params, minibatch) -> scalar loss
        data: Full dataset (X, Y)
        wstar_flat: (d,) flattened center parameters (w*)
        n_data: Number of data points
        beta: Inverse temperature (typically 1/log(n))
        gamma: Localizer strength
        batch_size: Minibatch size
        whitener: Geometry whitening transformation
        optimizer: Optax optimizer

    Returns:
        JIT-compiled step function: (key, params, state) -> ((params, state), metrics)
    """
    X, Y = data
    ref_dtype = wstar_flat.dtype

    # Precompute constants
    beta_tilde = jnp.asarray(beta * n_data, dtype=ref_dtype)
    gamma_val = jnp.asarray(gamma, dtype=ref_dtype)

    # Flatten/unflatten utilities (closure over wstar structure)
    def unflatten(flat: jnp.ndarray) -> Any:
        """Unflatten to match wstar structure."""
        # For now, assume wstar is already flat (single array)
        # In full implementation, would use tree_unflatten from wstar's treedef
        return flat

    @jax.jit
    def step_fn(
        key: jax.random.PRNGKey, params: VIParams, state: VIOptState
    ) -> Tuple[Tuple[VIParams, VIOptState], Dict[str, jnp.ndarray]]:
        """One ELBO optimization step."""

        # 1) Sample w ~ q (pathwise)
        w_flat, aux = sample_q(key, params, wstar_flat, whitener)
        w = unflatten(w_flat)
        tilde_v = aux["tilde_v"]
        m = aux["m"]
        z = aux["z"]
        eps = aux["eps"]

        # 2) Draw minibatch
        key_batch = jax.random.split(key)[0]
        idx = jax.random.choice(key_batch, X.shape[0], shape=(batch_size,), replace=True)
        minibatch = (X[idx], Y[idx])

        # 3) ELBO objective: ℓ(w) = -nβ L_batch(w) - ½γ ||tilde_v||^2
        # (STL: no explicit -log q term in gradients for continuous params)
        Ln_batch = loss_batch_fn(w, minibatch)  # Mean loss on batch
        localizer = 0.5 * gamma_val * jnp.dot(tilde_v, tilde_v)
        ell = -(beta_tilde * Ln_batch + localizer)

        # 4) Compute responsibilities and log q(w) for RB
        logps, r = logpdf_components_and_resp(tilde_v, params)
        logq = jax.nn.logsumexp(jax.nn.log_softmax(params.alpha) + logps)

        # RB payoff (centered by baseline)
        payoff = (ell - logq) - state.baseline

        # 5) Gradients:
        # - Continuous (rho, A): STL pathwise via w -> ell, stop-grad for log q
        # - Mixture logits (alpha): RB score gradient (r - pi) * payoff

        def ell_only(p: VIParams) -> jnp.ndarray:
            """Rebuild sample with new params, stop-grad for log q path."""
            rho, A, alpha = p  # noqa: F841 (alpha not used, but needed for signature)
            D_sqrt_new = softplus(rho)
            A_m = A[m]
            K_m_z = (D_sqrt_new[:, None] * A_m) @ z
            tilde_v_new = K_m_z + D_sqrt_new * eps
            w_new = unflatten(wstar_flat + whitener.from_tilde(tilde_v_new))

            Ln_b = loss_batch_fn(w_new, minibatch)
            loc = 0.5 * gamma_val * jnp.dot(tilde_v_new, tilde_v_new)
            return -(beta_tilde * Ln_b + loc)  # Maximize ell => minimize -ell

        # Continuous gradients via STL
        loss_val, grads_cont = jax.value_and_grad(ell_only)(params)

        # RB gradient for alpha
        pi = jax.nn.softmax(params.alpha)
        g_alpha = (r - pi) * payoff

        grads = VIParams(rho=grads_cont.rho, A=grads_cont.A, alpha=g_alpha)

        # 6) Optax update
        updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # 7) Update baseline (EMA)
        new_baseline = 0.99 * state.baseline + 0.01 * (ell - logq)

        new_state = VIOptState(
            opt_state=new_opt_state,
            baseline=new_baseline,
            step=state.step + jnp.array(1, dtype=jnp.int32),
        )

        # Metrics for tracing
        metrics = {
            "elbo_like": ell,  # ELBO-like term (no explicit entropy in STL path)
            "logq": logq,
            "radius2": jnp.dot(tilde_v, tilde_v),  # ||tilde_v||^2
            "Ln_batch": Ln_batch,  # Minibatch loss
            "work_fge": jnp.asarray(batch_size / float(n_data), dtype=jnp.float64),  # FGEs
        }

        return (new_params, new_state), metrics

    return step_fn
