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


# === HVP (Hessian-Vector Product) Utilities for Control Variate ===


def hvp_at_wstar(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    wstar: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Hessian-vector product H(w*) @ v using forward-over-reverse.

    Args:
        loss_fn: Loss function taking flattened parameters
        wstar: Point at which to evaluate Hessian (typically ERM solution)
        v: Direction vector (same shape as wstar)

    Returns:
        Hv: Hessian-vector product at wstar
    """
    # Forward-over-reverse: efficient for H @ v when H is not explicitly formed
    # jvp of grad gives us (∇L(w*), H(w*) @ v)
    grad_fn = jax.grad(loss_fn)
    _, hvp = jax.jvp(grad_fn, (wstar,), (v,))
    return hvp


def compute_subspace_hessian_diag(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    wstar: jnp.ndarray,
    params: "VIParams",
    pi: jnp.ndarray,
    whitener: Whitener,
) -> jnp.ndarray:
    """Compute diagonal of projected Hessian in the learned subspace.

    For each component m and each basis direction A_m[:,j], computes:
        v_mj = from_whitened(D^{1/2} A_m[:,j])
        diag_mj = v_mj^T H v_mj

    Args:
        loss_fn: Loss function (on flattened params)
        wstar: ERM solution (flattened)
        params: VI parameters with learned subspace
        pi: Mixture weights (M,)
        whitener: Whitening transformation

    Returns:
        hess_diag: (M, r) array where hess_diag[m,j] = v_mj^T H v_mj
    """
    from lambda_hat.variational import diag_from_rho

    D_sqrt, _ = diag_from_rho(params.rho)  # (d,)
    M, _, r = params.A.shape

    def compute_component_diag(m):
        """Compute Hessian diagonal for component m."""
        A_m = params.A[m]  # (d, r)

        def compute_direction_diag(j):
            """Compute v^T H v for direction j."""
            # Get whitened direction: tilde_v = D^{1/2} A_m[:,j]
            tilde_v = D_sqrt * A_m[:, j]  # (d,)
            # Map to model coordinates: v = from_whitened(tilde_v)
            v = whitener.from_tilde(tilde_v)  # (d,)
            # Compute Hv
            Hv = hvp_at_wstar(loss_fn, wstar, v)  # (d,)
            # Return v^T H v
            return jnp.dot(v, Hv)

        # Vmap over r directions
        return jax.vmap(compute_direction_diag)(jnp.arange(r))

    # Vmap over M components
    hess_diag = jax.vmap(compute_component_diag)(jnp.arange(M))  # (M, r)
    return hess_diag


def compute_trace_term(
    hess_diag: jnp.ndarray,
    pi: jnp.ndarray,
) -> jnp.ndarray:
    """Compute tr(H Sigma_q) using subspace approximation.

    For mixture of factor analyzers with Sigma_m = D + K_m K_m^T,
    the trace is approximately:
        tr(H Sigma_q) ≈ sum_m pi_m * sum_j (v_mj^T H v_mj)

    where v_mj are the learned basis directions.

    Args:
        hess_diag: (M, r) array of v^T H v values
        pi: (M,) mixture weights

    Returns:
        trace: scalar tr(H Sigma_q)
    """
    # Sum over directions within each component
    component_traces = jnp.sum(hess_diag, axis=1)  # (M,)
    # Weighted sum over components
    trace = jnp.dot(pi, component_traces)
    return trace


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
    unravel_fn: Callable,
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
        unravel_fn: Function to reconstruct PyTree from flat array
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
        """Unflatten to match wstar PyTree structure."""
        return unravel_fn(flat)

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
        Xb, Yb = X[idx], Y[idx]

        # 3) ELBO objective: ℓ(w) = -nβ L_batch(w) - ½γ ||tilde_v||^2
        # (STL: no explicit -log q term in gradients for continuous params)
        Ln_batch = loss_batch_fn(w, Xb, Yb)  # Mean loss on batch
        half = jnp.asarray(0.5, dtype=ref_dtype)
        localizer = half * gamma_val * jnp.dot(tilde_v, tilde_v)
        ell = jnp.asarray(-(beta_tilde * Ln_batch + localizer), dtype=ref_dtype)

        # 4) Compute responsibilities and log q(w) for RB
        logps, r = logpdf_components_and_resp(tilde_v, params)
        logq = jnp.asarray(
            jax.nn.logsumexp(jax.nn.log_softmax(params.alpha) + logps), dtype=ref_dtype
        )

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

            Ln_b = loss_batch_fn(w_new, Xb, Yb)
            loc = half * gamma_val * jnp.dot(tilde_v_new, tilde_v_new)
            return -(beta_tilde * Ln_b + loc)  # Maximize ell => minimize -ell

        # Continuous gradients via STL
        loss_val, grads_cont = jax.value_and_grad(ell_only)(params)

        # RB gradient for alpha
        pi = jax.nn.softmax(params.alpha)
        g_alpha = (r - pi) * payoff

        # Cast all gradients to ref_dtype to prevent float64 promotion
        grads = VIParams(
            rho=jnp.asarray(grads_cont.rho, dtype=ref_dtype),
            A=jnp.asarray(grads_cont.A, dtype=ref_dtype),
            alpha=jnp.asarray(g_alpha, dtype=ref_dtype),
        )

        # 6) Optax update
        updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Apply column normalization to A for numerical stability
        new_params = new_params._replace(A=jax.vmap(normalize_columns)(new_params.A))

        # 7) Update baseline (EMA)
        # Cast constants to ref_dtype to prevent float64 promotion
        decay = jnp.asarray(0.99, dtype=ref_dtype)
        rate = jnp.asarray(0.01, dtype=ref_dtype)
        baseline_update = decay * state.baseline + rate * (ell - logq)
        new_baseline = jnp.asarray(baseline_update, dtype=ref_dtype)

        new_state = VIOptState(
            opt_state=new_opt_state,
            baseline=new_baseline,
            step=state.step + jnp.array(1, dtype=jnp.int32),
        )

        # Metrics for tracing
        # Compute responsibility entropy: H(r) = -sum(r * log(r))
        # Use numerically stable formulation: where r=0, contribution is 0
        resp_entropy = -jnp.sum(jnp.where(r > 1e-10, r * jnp.log(r + 1e-10), 0.0))

        metrics = {
            "elbo": ell + logq,  # TRUE ELBO (target + entropy)
            "elbo_like": ell,  # ELBO-like term (target only, for debugging)
            "logq": logq,
            "radius2": jnp.dot(tilde_v, tilde_v),  # ||tilde_v||^2
            "Ln_batch": Ln_batch,  # Minibatch loss
            "resp_entropy": resp_entropy,  # Entropy of responsibilities (detects peaking)
            "work_fge": jnp.asarray(batch_size / float(n_data), dtype=jnp.float64),  # FGEs
        }

        return (new_params, new_state), metrics

    return step_fn


def apply_control_variate(
    loss_samples: jnp.ndarray,
    perturbations: jnp.ndarray,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    wstar: jnp.ndarray,
    hess_diag: jnp.ndarray,
    pi: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Apply HVP-based control variate to reduce variance of E_q[L] estimate.

    Uses the identity:
        E_q[L] = E_q[L - 0.5 * v^T H v] + 0.5 * tr(H Sigma_q)

    Args:
        loss_samples: (S,) array of L(w* + v_s) values
        perturbations: (S, d) array of v_s samples from q
        loss_fn: Loss function on flattened params
        wstar: ERM solution (d,)
        hess_diag: (M, r) array of v_mj^T H v_mj values from subspace
        pi: (M,) mixture weights

    Returns:
        Eq_Ln_cv: Control-variate corrected estimate of E_q[L]
        cv_info: Dict with intermediate values for diagnostics
    """

    # Compute 0.5 * v^T H v for each sample
    def compute_quadratic(v):
        Hv = hvp_at_wstar(loss_fn, wstar, v)
        return 0.5 * jnp.dot(v, Hv)

    quadratics = jax.vmap(compute_quadratic)(perturbations)  # (S,)

    # Apply control variate: L - 0.5 * v^T H v
    cv_samples = loss_samples - quadratics  # (S,)
    Eq_Ln_cv_raw = jnp.mean(cv_samples)

    # Add trace correction: 0.5 * tr(H Sigma_q)
    trace_term = 0.5 * compute_trace_term(hess_diag, pi)
    Eq_Ln_cv = Eq_Ln_cv_raw + trace_term

    # Diagnostics
    cv_var = jnp.var(cv_samples)
    cv_info = {
        "Eq_Ln_mc": jnp.mean(loss_samples),  # Raw MC estimate (no CV)
        "Eq_Ln_cv": Eq_Ln_cv,  # CV-corrected estimate
        "trace_term": trace_term,  # 0.5 * tr(H Sigma_q)
        "mean_quadratic": jnp.mean(quadratics),  # Mean of 0.5 * v^T H v
        "variance_reduction": jnp.where(
            cv_var > 1e-10,
            jnp.var(loss_samples) / cv_var,
            jnp.array(1.0),
        ),  # Variance reduction factor
    }

    return Eq_Ln_cv, cv_info


# === High-Level API ===


def fit_vi_and_estimate_lambda(
    rng_key: jax.random.PRNGKey,
    loss_batch_fn: Callable,
    loss_full_fn: Callable,
    wstar_flat: jnp.ndarray,
    unravel_fn: Callable,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    n_data: int,
    beta: float,
    gamma: float,
    M: int,
    r: int,
    steps: int,
    batch_size: int,
    lr: float,
    eval_samples: int,
    whitener: Whitener,
) -> Tuple[float, Dict[str, jnp.ndarray], Dict[str, Any]]:
    """Fit variational distribution and estimate Local Learning Coefficient.

    Runs ELBO optimization, then performs final MC estimation on full dataset.

    Args:
        rng_key: JRNG key
        loss_batch_fn: Function (params, minibatch) -> scalar loss
        loss_full_fn: Function (params) -> scalar loss on full dataset
        wstar_flat: (d,) flattened ERM parameters (center of local posterior)
        unravel_fn: Function to reconstruct PyTree from flat array
        data: Tuple of (X, Y)
        n_data: Number of data points
        beta: Inverse temperature (typically 1/log(n))
        gamma: Localizer strength
        M: Number of mixture components
        r: Rank budget per component
        steps: Total optimization steps
        batch_size: Minibatch size
        lr: Learning rate for Adam optimizer
        eval_samples: Number of MC samples for final LLC estimate
        whitener: Geometry whitening transformation

    Returns:
        lambda_hat: LLC estimate λ̂_VI = nβ(E_q[L_n] - L_n(w*))
        traces: Dict of optimization traces (for analysis)
        extras: Dict with {π, D_sqrt, Eq_Ln, Ln_wstar, final_params}
    """
    key_init, key_opt, key_eval = jax.random.split(rng_key, 3)
    ref_dtype = wstar_flat.dtype

    # Initialize VI parameters and optimizer
    vi_params = init_vi_params(key_init, wstar_flat, M=M, r=r)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(vi_params)
    vi_state = VIOptState(
        opt_state=opt_state,
        baseline=jnp.array(0.0, dtype=ref_dtype),
        step=jnp.array(0, dtype=jnp.int32),
    )

    # Build ELBO step function
    step_fn = build_elbo_step(
        loss_batch_fn=loss_batch_fn,
        data=data,
        wstar_flat=wstar_flat,
        unravel_fn=unravel_fn,
        n_data=n_data,
        beta=beta,
        gamma=gamma,
        batch_size=batch_size,
        whitener=whitener,
        optimizer=optimizer,
    )

    # Run optimization loop
    def scan_body(carry, step_idx):
        params, state, cumulative_fge = carry
        step_key = jax.random.fold_in(key_opt, step_idx)
        (new_params, new_state), metrics = step_fn(step_key, params, state)
        new_fge = cumulative_fge + metrics["work_fge"]
        # Add cumulative FGE to metrics for tracing
        metrics_with_fge = {**metrics, "cumulative_fge": new_fge}
        return (new_params, new_state, new_fge), metrics_with_fge

    init_carry = (vi_params, vi_state, jnp.array(0.0, dtype=jnp.float64))
    step_indices = jnp.arange(steps)
    (final_params, final_state, _), trace_metrics = jax.lax.scan(
        scan_body, init_carry, step_indices
    )

    # Extract final variational parameters
    rho_final, A_final, alpha_final = final_params
    D_sqrt_final, _ = diag_from_rho(rho_final)
    pi_final = jax.nn.softmax(alpha_final)

    # Flatten/unflatten utilities
    def unflatten(flat: jnp.ndarray) -> Any:
        """Unflatten to match wstar PyTree structure."""
        return unravel_fn(flat)

    # Final MC estimation on FULL dataset with HVP control variate
    eval_keys = jax.random.split(key_eval, eval_samples)

    # Sample perturbations and compute losses
    def sample_perturbation_and_loss(eval_key):
        """Sample from trained q and evaluate full-dataset loss."""
        w_flat, aux = sample_q(eval_key, final_params, wstar_flat, whitener)
        # Get perturbation in model coordinates: v = w - w*
        # Cast to match wstar_flat dtype
        v = jnp.asarray(w_flat - wstar_flat, dtype=wstar_flat.dtype)
        w = unflatten(w_flat)
        loss = loss_full_fn(w)
        return v, loss

    # Vectorize over eval_samples
    perturbations, Ln_samples = jax.vmap(sample_perturbation_and_loss)(eval_keys)

    # Compute L_n(w*)
    Ln_wstar = loss_full_fn(unflatten(wstar_flat))

    # Compute Hessian diagonal in learned subspace for control variate
    # Create loss function that takes flat params
    def loss_flat(w_flat):
        return loss_full_fn(unflatten(w_flat))

    hess_diag = compute_subspace_hessian_diag(
        loss_fn=loss_flat, wstar=wstar_flat, params=final_params, pi=pi_final, whitener=whitener
    )

    # Apply control variate to get improved estimate
    Eq_Ln_cv, cv_info = apply_control_variate(
        loss_samples=Ln_samples,
        perturbations=perturbations,
        loss_fn=loss_flat,
        wstar=wstar_flat,
        hess_diag=hess_diag,
        pi=pi_final,
    )

    # Use CV-corrected estimate for lambda_hat (can also return raw MC for comparison)
    Eq_Ln = Eq_Ln_cv  # Use CV-corrected estimate
    lambda_hat = n_data * beta * (Eq_Ln - Ln_wstar)

    # Package results
    traces = {
        "elbo": trace_metrics["elbo"],
        "elbo_like": trace_metrics["elbo_like"],  # Target term only (for debugging)
        "logq": trace_metrics["logq"],
        "radius2": trace_metrics["radius2"],
        "Ln_batch": trace_metrics["Ln_batch"],
        "resp_entropy": trace_metrics["resp_entropy"],  # Responsibility entropy (detects peaking)
        "cumulative_fge": trace_metrics["cumulative_fge"],
    }

    extras = {
        "pi": pi_final,
        "D_sqrt": D_sqrt_final,
        "Eq_Ln": Eq_Ln,  # CV-corrected estimate
        "Eq_Ln_mc": cv_info["Eq_Ln_mc"],  # Raw MC estimate (for comparison)
        "Eq_Ln_cv": Eq_Ln_cv,  # CV-corrected estimate (same as Eq_Ln)
        "Ln_wstar": Ln_wstar,
        "Ln_samples": Ln_samples,
        "eval_samples": eval_samples,
        "final_params": final_params,
        "cv_info": cv_info,  # Full CV diagnostics
    }

    return lambda_hat, traces, extras
